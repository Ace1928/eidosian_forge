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
def call_setattr(self, tx, obj: VariableTracker, name_var: VariableTracker, val: VariableTracker):
    from .distributed import PlacementVariable
    if isinstance(obj, (variables.DataClassVariable, variables.CustomizedDictVariable, PlacementVariable)):
        return obj.call_method(tx, '__setattr__', [name_var, val], {})
    elif tx.output.side_effects.is_attribute_mutation(obj) and name_var.is_python_constant():
        name = name_var.as_python_constant()
        if isinstance(obj, variables.TensorVariable):
            from .builder import wrap_fx_proxy
            if name == 'requires_grad':
                unimplemented('mutating requires_grad can introduce a new leaf from non-leaf or vice versa in the middle of the graph, which aot_autograd does not currently know how to handle. ')
            if name == 'data':
                to_remove = []
                for tf in tx.output.tracked_fakes:
                    if tf.source == obj.source:
                        to_remove.append(tf)
                for tf in to_remove:
                    tx.output.tracked_fakes.remove(tf)
                with dynamo_disable_grad(tx), torch.no_grad():
                    out = wrap_fx_proxy(tx, tx.output.create_proxy('call_function', torch.Tensor.set_, *proxy_args_kwargs([obj, val], {})))

                def _lower_version_count_by_1(x):
                    version = x._version
                    if version > 0:
                        version = version - 1
                    torch._C._autograd._unsafe_set_version_counter(x, version)
                    return x
                tx.output.create_proxy('call_function', _lower_version_count_by_1, (out.as_proxy(),), {})
                _lower_version_count_by_1(obj.as_proxy().node.meta['example_value'])
                return out
        tx.output.side_effects.store_attr(obj, name, val)
        return val
    elif isinstance(obj, variables.UserDefinedObjectVariable):
        unimplemented(f'setattr(UserDefinedObjectVariable) {type(obj.value).__setattr__}')
    elif isinstance(obj, variables.NNModuleVariable):
        if not tx.output.is_root_tracer():
            raise AttributeMutationError("Can't inplace modify module params/buffers inside HigherOrderOp")
        if name_var.is_python_constant() and isinstance(val, variables.TensorVariable):
            assigning_fake_val = get_fake_value(val.as_proxy().node, tx)
            try:
                getattr_var = obj.var_getattr(tx, name_var.as_python_constant())
            except AttributeError:
                getattr_var = None
            if isinstance(getattr_var, variables.TensorVariable):
                existing_fake_attr = get_fake_value(getattr_var.as_proxy().node, tx)
                mod_setattr = inspect.getattr_static(obj.module_type, '__setattr__')
                if existing_fake_attr is assigning_fake_val and mod_setattr is torch.nn.Module.__setattr__:
                    return getattr_var
        obj.convert_to_unspecialized(tx)
    elif isinstance(obj, variables.dicts.HFPretrainedConfigVariable) and tx.export:
        if name_var.is_python_constant() and isinstance(val, variables.ConstantVariable):
            setattr(obj.obj, name_var.as_python_constant(), val.as_python_constant())
            return ConstantVariable(None)