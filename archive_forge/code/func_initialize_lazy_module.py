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
def initialize_lazy_module(tx, mod, args, kwargs):
    """
    Fairly coupled helper used by NNModuleVariable and UnspecializedNNModuleVariable.

    Used to cause lazy module to be initialized (and delete its init hook) before tracing. Especially
    useful now that 'allowed' modules graph-break on hooks, calling this first ensures there is no hook
    by the time we trace __call__ and thus no graph-break for lazy allowed modules.
    """
    assert len(kwargs) == 0
    if hasattr(mod, '_initialize_hook'):

        def convert_to_fake(x):
            if isinstance(x, torch.fx.Proxy):
                return get_fake_value(x.node, tx)
            else:
                return x
        input = [type(arg)([convert_to_fake(x) for x in arg]) if isinstance(arg, (list, tuple)) else convert_to_fake(arg) for arg in proxy_args_kwargs(args, {})[0]]
        mod._infer_parameters(mod, input)