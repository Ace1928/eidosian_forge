from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, TypeVar, Union
import torch
from typing_extensions import override
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from lightning_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
def measure_flops(model: torch.nn.Module, forward_fn: Callable[[], torch.Tensor], loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]=None) -> int:
    """Utility to compute the total number of FLOPs used by a module during training or during inference.

    It's recommended to create a meta-device model for this:

    Example::

        with torch.device("meta"):
            model = MyModel()
            x = torch.randn(2, 32)

        model_fwd = lambda: model(x)
        fwd_flops = measure_flops(model, model_fwd)

        model_loss = lambda y: y.sum()
        fwd_and_bwd_flops = measure_flops(model, model_fwd, model_loss)

    Args:
        model: The model whose FLOPs should be measured.
        forward_fn: A function that runs ``forward`` on the model and returns the result.
        loss_fn: A function that computes the loss given the ``forward_fn`` output. If provided, the loss and `backward`
            FLOPs will be included in the result.

    """
    if not _TORCH_GREATER_EQUAL_2_1:
        raise ImportError('`measure_flops` requires PyTorch >= 2.1.')
    from torch.utils.flop_counter import FlopCounterMode
    flop_counter = FlopCounterMode(model, display=False)
    with flop_counter:
        if loss_fn is None:
            forward_fn()
        else:
            loss_fn(forward_fn()).backward()
    return flop_counter.get_total_flops()