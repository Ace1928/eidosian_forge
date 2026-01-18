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
def setup_globally_cached(self, name, value, push_null):
    """Store value in a new global"""
    name = re.sub('[^a-zA-Z0-9_]+', '_', name)
    f_globals = self.tx.f_globals
    if name in f_globals:
        assert id(f_globals[name]) == id(value)
    else:
        f_globals[name] = value
    return [self.create_load_global(name, push_null, add=True)]