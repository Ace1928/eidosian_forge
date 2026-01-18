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
@staticmethod
def call_custom_dict_fromkeys(tx, user_cls, *args, **kwargs):
    assert user_cls in {dict, OrderedDict, defaultdict}
    if kwargs:
        assert user_cls is OrderedDict
        assert len(args) == 1 and len(kwargs) == 1 and ('value' in kwargs)
        args = (*args, kwargs.pop('value'))
    if len(args) == 0:
        raise UserError(TypeError, 'fromkeys expected at least 1 argument, got 0')
    if len(args) == 1:
        args = (*args, ConstantVariable.create(None))
    assert len(args) == 2
    arg, value = args
    DictVariableType = ConstDictVariable if user_cls is not defaultdict else DefaultDictVariable
    if isinstance(arg, dict):
        return DictVariableType(dict.fromkeys(arg, value), user_cls, mutable_local=MutableLocal())
    elif isinstance(arg, (ConstDictVariable, ListVariable, TupleVariable, ListIteratorVariable)):
        keys = [DictVariableType.get_key(x) for x in arg.unpack_var_sequence(tx)]
        return DictVariableType(dict.fromkeys(keys, value), user_cls, mutable_local=MutableLocal())
    unimplemented(f'{user_cls.__name__}.fromkeys(): {args} {kwargs}')