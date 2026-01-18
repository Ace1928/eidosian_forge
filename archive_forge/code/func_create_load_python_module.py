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
def create_load_python_module(self, mod, push_null):
    """
        Generate a LOAD_GLOBAL instruction to fetch a given python module.
        """
    global_scope = self.tx.output.global_scope
    name = re.sub('^.*[.]', '', mod.__name__)
    if global_scope.get(name, None) is mod:
        return self.create_load_global(name, push_null, add=True)
    mangled_name = f'___module_{name}_{id(mod)}'
    if mangled_name not in global_scope:
        self.tx.output.install_global(mangled_name, mod)
    return self.create_load_global(mangled_name, push_null, add=True)