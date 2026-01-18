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
def gen_source(source, name):
    name_split = name.split('.')
    if name_split[0] == '':
        return source
    while len(name_split) > 0:
        x = name_split.pop(0)
        source = AttrSource(source, x)
    return source