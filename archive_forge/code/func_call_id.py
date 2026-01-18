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
def call_id(self, tx, *args):
    if len(args) > 0 and isinstance(args[0], variables.NNModuleVariable):
        nn_mod_variable = args[0]
        mod = tx.output.get_submodule(nn_mod_variable.module_key)
        return variables.ConstantVariable.create(id(mod))
    else:
        unimplemented(f'call_id with args {args}')