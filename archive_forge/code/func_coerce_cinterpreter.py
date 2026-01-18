from abc import ABC, abstractmethod
import contextlib
from typing import Any
import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
from torch.autograd.forward_ad import _set_fwd_grad_enabled
def coerce_cinterpreter(cinterpreter: CInterpreter) -> FuncTorchInterpreter:
    key = cinterpreter.key()
    if key == TransformType.Grad:
        return GradInterpreter(cinterpreter)
    if key == TransformType.Vmap:
        return VmapInterpreter(cinterpreter)
    if key == TransformType.Jvp:
        return JvpInterpreter(cinterpreter)
    if key == TransformType.Functionalize:
        return FunctionalizeInterpreter(cinterpreter)
    raise RuntimeError(f'NYI: PyDispatcher has not implemented support for {key}')