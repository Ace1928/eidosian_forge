import collections
import functools
import inspect
import sys
import textwrap
import types
import warnings
from typing import Dict, List, Set, Type
import torch
import torch._jit_internal as _jit_internal
from torch._sources import fake_range
from torch.jit._builtins import _find_builtin
from torch.jit._check import AttributeTypeIsSupportedChecker
from torch.jit._state import _add_script_class, _get_script_class, _python_cu
from torch.jit.frontend import (
from torch.nn import Module
def create_hooks_from_stubs(concrete_type, hook_stubs, pre_hook_stubs):
    hook_defs = [h.def_ for h in hook_stubs]
    hook_rcbs = [h.resolution_callback for h in hook_stubs]
    pre_hook_defs = [h.def_ for h in pre_hook_stubs]
    pre_hook_rcbs = [h.resolution_callback for h in pre_hook_stubs]
    concrete_type._create_hooks(hook_defs, hook_rcbs, pre_hook_defs, pre_hook_rcbs)