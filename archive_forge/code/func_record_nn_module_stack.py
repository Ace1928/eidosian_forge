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
@contextmanager
def record_nn_module_stack(module_key: str, source, tx, mod: torch.nn.Module):
    fully_qualified_name = source.name()
    try:
        tx.nn_module_stack[module_key] = (fully_qualified_name, type(mod))
        yield
    finally:
        del tx.nn_module_stack[module_key]