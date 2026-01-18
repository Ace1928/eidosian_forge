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
def create_call_function_kw(self, nargs, kw_names, push_null) -> List[Instruction]:
    if sys.version_info >= (3, 11):
        output = create_call_function(nargs, push_null)
        assert output[-2].opname == 'PRECALL'
        kw_names_inst = create_instruction('KW_NAMES', argval=kw_names)
        output.insert(-2, kw_names_inst)
        return output
    return [self.create_load_const(kw_names), create_instruction('CALL_FUNCTION_KW', arg=nargs)]