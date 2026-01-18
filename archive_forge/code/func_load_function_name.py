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
def load_function_name(self, fn_name, push_null, num_on_stack=0):
    """Load the global fn_name on the stack num_on_stack down"""
    output = []
    if push_null and sys.version_info >= (3, 11):
        output.extend([create_instruction('PUSH_NULL'), *self.rot_n(num_on_stack + 1)])
    output.extend([self.create_load_global(fn_name, False, add=True), *self.rot_n(num_on_stack + 1)])
    return output