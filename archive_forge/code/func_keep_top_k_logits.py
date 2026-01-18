import math
from typing import Callable, Optional, Protocol, Tuple
import torch
def keep_top_k_logits(k: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a function that masks logits values smaller than the top `k` ones.

    Parameters
    ----------
    k
        The ranking below which logit values are replaced by `-math.inf`.

    """
    if not isinstance(k, int) or k < 1:
        raise ValueError(f'`k` must be a strictly positive integers, got {k} instead.')

    def logits_processor(logits: torch.Tensor) -> torch.Tensor:
        num_to_keep = min(k, logits.size(-1))
        mask_idx = logits < torch.topk(logits, num_to_keep)[0][..., -1, None]
        return logits.masked_fill(mask_idx, -math.inf)
    return logits_processor