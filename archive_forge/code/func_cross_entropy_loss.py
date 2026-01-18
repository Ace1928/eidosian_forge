from typing import Tuple, Optional, Union
import torch
from einops import rearrange
import triton
import triton.language as tl
def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float=0.0, logit_scale: float=1.0, lse_square_scale: float=0.0, ignored_index=-100, inplace_backward: bool=False, process_group=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        logits: (batch, vocab_size)
        labels: (batch,)
        label_smoothing: float
        logit_scale: float. Multiply logits by this scale before calculating the loss.
        lse_square_scale: float. If > 0, we add lse_square_scale * lse(logits) ^ 2 to the loss.
            This is also referred to as "z-loss".
        ignored_index: int. If labels == ignored_index, the loss is set to 0.0.
        inplace_backward: bool. If True, we do the backward pass in-place by modifying the logits.
            This saves memory.
        process_group: if not None, we're doing Tensor Parallel: each process is responsible for
            one part of the vocab. The loss will be aggregated across processes.
    Returns:
        losses: (batch,), float
        z_losses: (batch,), float
    """
    return CrossEntropyLoss.apply(logits, labels, label_smoothing, logit_scale, lse_square_scale, ignored_index, inplace_backward, process_group)