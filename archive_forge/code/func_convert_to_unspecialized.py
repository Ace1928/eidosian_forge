import functools
import inspect
import itertools
import types
from contextlib import contextmanager, nullcontext
from typing import Dict, List
import torch.nn
from .. import skipfiles, variables
from ..allowed_functions import is_allowed
from ..exc import unimplemented, UnspecializeRestartAnalysis, Unsupported
from ..guards import GuardBuilder, install_guard
from ..mutation_guard import GenerationTracker
from ..source import (
from ..utils import (
from .base import MutableLocal, typestr, VariableTracker
from .functions import invoke_and_store_as_constant
from .lists import SliceVariable
from .user_defined import UserDefinedObjectVariable
def convert_to_unspecialized(self, tx):
    """Restart analysis treating this module as an UnspecializedNNModuleVariable"""
    mod = tx.output.get_submodule(self.module_key)
    GenerationTracker.tag(mod)
    if tx.f_code.co_name != '__init__':
        GenerationTracker.mark_class_dynamic(type(mod))
    raise UnspecializeRestartAnalysis()