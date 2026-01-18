import functools
import torch
import torch.distributed as dist
from typing import Optional
def reduce_scatter_hook(state: DefaultState, grad: torch.Tensor, output: torch.Tensor):
    """
    Implement the  FSDP communication hook for ``reduce_scatter`` algorithm.

    For sharded FSDP strategies and a necessary pre- and post-division of gradients.

    Args:
        state (DefaultState): State information, configures pre- and post-division factors.
        grad (torch.Tensor): An unsharded gradient for the local batch that needs to be
        communicated across ranks.
        output (torch.Tensor): Stores a single shard of the gradient after ``reduce_scatter``.
    """
    if state.gradient_predivide_factor > 1:
        grad.div_(state.gradient_predivide_factor)
    dist.reduce_scatter_tensor(output, grad, group=state.process_group)
    if state.gradient_postdivide_factor > 1:
        output.div_(state.gradient_postdivide_factor)