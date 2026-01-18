from typing import Any, Callable, Dict, Optional, Tuple, cast
from ..compat import torch
from ..config import registry
from ..model import Model
from ..shims import PyTorchGradScaler, PyTorchShim
from ..types import ArgsKwargs, Floats3d, Padded
from ..util import (
def convert_pytorch_default_inputs(model: Model, X: Any, is_train: bool) -> Tuple[ArgsKwargs, Callable[[ArgsKwargs], Any]]:
    shim = cast(PyTorchShim, model.shims[0])
    xp2torch_ = lambda x: xp2torch(x, requires_grad=is_train, device=shim.device)
    converted = convert_recursive(is_xp_array, xp2torch_, X)
    if isinstance(converted, ArgsKwargs):

        def reverse_conversion(dXtorch):
            return convert_recursive(is_torch_array, torch2xp, dXtorch)
        return (converted, reverse_conversion)
    elif isinstance(converted, dict):

        def reverse_conversion(dXtorch):
            dX = convert_recursive(is_torch_array, torch2xp, dXtorch)
            return dX.kwargs
        return (ArgsKwargs(args=tuple(), kwargs=converted), reverse_conversion)
    elif isinstance(converted, (tuple, list)):

        def reverse_conversion(dXtorch):
            dX = convert_recursive(is_torch_array, torch2xp, dXtorch)
            return dX.args
        return (ArgsKwargs(args=tuple(converted), kwargs={}), reverse_conversion)
    else:

        def reverse_conversion(dXtorch):
            dX = convert_recursive(is_torch_array, torch2xp, dXtorch)
            return dX.args[0]
        return (ArgsKwargs(args=(converted,), kwargs={}), reverse_conversion)