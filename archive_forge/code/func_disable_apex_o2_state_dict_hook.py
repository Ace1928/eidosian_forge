from __future__ import annotations
import contextlib
import copy
import inspect
import io
import re
import textwrap
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
import torch.jit._trace
import torch.serialization
from torch import _C
from torch.onnx import (  # noqa: F401
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import (
@contextlib.contextmanager
@_beartype.beartype
def disable_apex_o2_state_dict_hook(model: Union[torch.nn.Module, torch.jit.ScriptFunction]):
    if not isinstance(model, torch.jit.ScriptFunction):
        model_hooks = {}
        for module in model.modules():
            for key, hook in module._state_dict_hooks.items():
                if type(hook).__name__ == 'O2StateDictHook':
                    if module not in model_hooks:
                        model_hooks[module] = {}
                    model_hooks[module][key] = hook
            if module in model_hooks:
                for key in model_hooks[module]:
                    module._state_dict_hooks.pop(key)
        try:
            yield
        finally:
            for module, m_map in model_hooks.items():
                for key, hook in m_map.items():
                    module._state_dict_hooks[key] = hook
    else:
        try:
            yield
        finally:
            pass