import contextlib
import functools
import inspect
import itertools
import logging
import math
import operator
import types
from collections import defaultdict, OrderedDict
from typing import Dict, List
import torch
from torch import sym_float, sym_int
from .. import config, polyfill, variables
from ..exc import (
from ..guards import GuardBuilder, install_guard
from ..replay_record import DummyModule
from ..source import AttrSource, GetItemSource, is_constant_source, TypeSource
from ..utils import (
from .base import MutableLocal, typestr, VariableTracker
from .constant import ConstantVariable
from .ctx_manager import EventVariable, StreamVariable
from .dicts import ConstDictVariable, DefaultDictVariable, SetVariable
from .lists import (
from .tensor import FakeItemVariable, SymNodeVariable, UnspecializedPythonVariable
from .user_defined import UserDefinedVariable
def call_getattr(self, tx, obj: VariableTracker, name_var: VariableTracker, default=None):
    from .. import trace_rules
    from . import ConstantVariable, GetAttrVariable, PythonModuleVariable, TorchInGraphFunctionVariable, TorchVariable, UserFunctionVariable
    from .builder import SourcelessBuilder, VariableBuilder
    name = name_var.as_python_constant()
    if not name_var.is_python_constant():
        unimplemented('non-const getattr() name')
    if tx.output.side_effects.is_attribute_mutation(obj):
        try:
            return tx.output.side_effects.load_attr(obj, name)
        except KeyError:
            pass
    if default is not None:
        hasattr_var = self.call_hasattr(tx, obj, name_var)
        assert hasattr_var.as_python_constant() in (True, False)
        if not hasattr_var.as_python_constant():
            return default
    options = {}
    if obj.source:
        source = AttrSource(obj.source, name)
        options['source'] = source
    else:
        source = None
    if name == '__bases__':
        try:
            value = obj.as_python_constant()
            if isinstance(value, type):
                bases = value.__bases__
                if source is not None:
                    tuple_args = [VariableBuilder(tx, GetItemSource(source, i))(b) for i, b in enumerate(bases)]
                else:
                    tuple_args = [SourcelessBuilder()(tx, b) for b in bases]
                return variables.TupleVariable(tuple_args, **options)
        except NotImplementedError:
            pass
    if isinstance(obj, variables.NNModuleVariable):
        return obj.var_getattr(tx, name)
    elif isinstance(obj, variables.TensorVariable) and name == 'grad':
        if source:
            for grapharg in tx.output.graphargs:
                if grapharg.source == source.base:
                    old_grad = grapharg.example.grad
                    new_grad = obj.as_proxy().node.meta['example_value'].grad

                    def _grad_changed(old, new):
                        if old is None or new is None:
                            return new is not old
                        try:
                            if old.shape != new.shape:
                                return True
                            if old.stride() != new.stride():
                                return True
                            return False
                        except TypeError as te:
                            unimplemented(str(te))
                    if _grad_changed(old_grad, new_grad):
                        if new_grad is not None:
                            grad_shape_specialized = [int(x) for x in new_grad.shape]
                            grapharg.example.grad = torch.zeros(grad_shape_specialized, device=new_grad.device)
                        else:
                            grapharg.example.grad = None
                    return VariableBuilder(tx, source)(grapharg.example.grad)
            unimplemented('tensor grad')
        else:
            unimplemented('tensor grad')
    elif isinstance(obj, (variables.TensorVariable, variables.NamedTupleVariable, variables.ConstantVariable, variables.UserDefinedClassVariable, variables.UserDefinedObjectVariable)):
        try:
            return obj.var_getattr(tx, name).clone(source=source)
        except NotImplementedError:
            return GetAttrVariable(obj, name, **options)
    elif isinstance(obj, TorchInGraphFunctionVariable):
        member = getattr(obj.value, name)
        if trace_rules.lookup(member) is not None:
            return trace_rules.lookup(member)(member, **options)
    elif isinstance(obj, TorchVariable):
        member = getattr(obj.value, name)
        if is_utils_checkpoint(member):
            options['source'] = source
            return build_checkpoint_variable(**options)
        elif trace_rules.lookup(member) is not None:
            return trace_rules.lookup(member)(member, **options)
        elif source is not None:
            return VariableBuilder(tx, source)(member)
        else:
            return SourcelessBuilder()(tx, member)
    elif isinstance(obj, (PythonModuleVariable, DummyModule)):
        member = obj.value.__dict__[name]
        if config.replay_record_enabled:
            tx.exec_recorder.record_module_access(obj.value, name, member)
        return VariableBuilder(tx, source)(member)
    elif istype(obj, UserFunctionVariable) and name in ('__name__', '__module__'):
        return ConstantVariable.create(getattr(obj.fn, name))
    else:
        try:
            return obj.var_getattr(tx, name).clone(source=source)
        except NotImplementedError:
            return GetAttrVariable(obj, name, **options)