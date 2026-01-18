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
def create_load_global(self, name, push_null, add=False):
    if add:
        self.tx.output.update_co_names(name)
    assert name in self.code_options['co_names'], f'{name} not in co_names'
    return create_load_global(name, push_null)