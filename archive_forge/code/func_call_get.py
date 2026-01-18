import collections
import dataclasses
import functools
import inspect
import sys
from typing import Any, Dict, List, Optional
import torch
import torch.fx
from .. import variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..eval_frame import skip_code
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard, make_dupe_guard
from ..source import AttrSource, GetItemSource, GlobalWeakRefSource
from ..utils import global_key_name, istensor, istype, iter_contains
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .tensor import TensorVariable
def call_get(self, tx, key: VariableTracker, default: Optional[VariableTracker]=None):
    from .builder import VariableBuilder
    k, has_key = self._contains_helper(tx, key)
    if has_key:
        return VariableBuilder(tx, GetItemSource(self.source, k))(sys.modules[k])
    if default is not None:
        return default
    return ConstantVariable.create(value=None)