from typing import Any, Callable, Dict, Optional, Tuple, cast
from ..compat import torch
from ..config import registry
from ..model import Model
from ..shims import PyTorchGradScaler, PyTorchShim
from ..types import ArgsKwargs, Floats3d, Padded
from ..util import (
def convert_pytorch_default_outputs(model: Model, X_Ytorch: Any, is_train: bool):
    shim = cast(PyTorchShim, model.shims[0])
    X, Ytorch = X_Ytorch
    Y = convert_recursive(is_torch_array, torch2xp, Ytorch)

    def reverse_conversion(dY: Any) -> ArgsKwargs:
        dYtorch = convert_recursive(is_xp_array, partial(xp2torch, device=shim.device), dY)
        return ArgsKwargs(args=((Ytorch,),), kwargs={'grad_tensors': dYtorch})
    return (Y, reverse_conversion)