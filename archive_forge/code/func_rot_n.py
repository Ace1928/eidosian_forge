import collections
import dataclasses
import re
import sys
import types
from typing import Counter, Dict, List, Optional
import torch.nn
from . import utils
from .bytecode_transformation import (
from .exc import unimplemented
from .source import AttrSource, Source
from .utils import is_safe_constant, rot_n_helper
from .variables.base import VariableTracker
from .variables.nn_module import NNModuleVariable
from .variables.tensor import (
from .variables.torch_function import TensorWithTFOverrideVariable
def rot_n(self, n):
    try:
        return create_rot_n(n)
    except AttributeError:
        return [create_instruction('BUILD_TUPLE', arg=n), self._create_load_const(rot_n_helper(n)), *create_rot_n(2), create_instruction('CALL_FUNCTION_EX', arg=0), create_instruction('UNPACK_SEQUENCE', arg=n)]