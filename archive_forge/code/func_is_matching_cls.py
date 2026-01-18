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
@staticmethod
def is_matching_cls(cls):
    mod = sys.modules.get('transformers.configuration_utils')
    is_match = mod is not None and issubclass(cls, mod.PretrainedConfig)
    if is_match:
        _install_PretrainedConfig_patch()
    return is_match