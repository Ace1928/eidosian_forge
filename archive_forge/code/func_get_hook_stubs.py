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
def get_hook_stubs(nn_module):
    """Return forward hook and pre_hook ScriptModuleStubs."""
    check_module_initialized(nn_module)
    hook_map: Dict = {}
    hook_stubs = []
    for hook in nn_module._forward_hooks.values():
        if hook.__name__ in hook_map:
            if id(hook) != id(hook_map[hook.__name__]):
                raise RuntimeError(f"Hook '{hook.__name__}' on {type(nn_module).__name__} has at least two different python definitions. Please use unique names for all hooks.")
        else:
            hook_map[hook.__name__] = hook
        hook_stubs.append(make_stub(hook, hook.__name__))
    pre_hook_stubs = []
    for pre_hook in nn_module._forward_pre_hooks.values():
        if pre_hook.__name__ in hook_map:
            if id(pre_hook) != id(hook_map[pre_hook.__name__]):
                raise RuntimeError(f"Pre-hook '{pre_hook.__name__}' on {type(nn_module).__name__} has at least two different python definitions. Please use unique names for all hooks.")
        else:
            hook_map[pre_hook.__name__] = pre_hook
        pre_hook_stubs.append(make_stub(pre_hook, pre_hook.__name__))
    return (hook_stubs, pre_hook_stubs)