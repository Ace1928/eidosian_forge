import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torchmetrics.utilities.exceptions import TorchMetricsUserWarning
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_12, _TORCH_GREATER_EQUAL_1_13, _XLA_AVAILABLE
from torchmetrics.utilities.prints import rank_zero_warn
def select_topk(prob_tensor: Tensor, topk: int=1, dim: int=1) -> Tensor:
    """Convert a probability tensor to binary by selecting top-k the highest entries.

    Args:
        prob_tensor: dense tensor of shape ``[..., C, ...]``, where ``C`` is in the
            position defined by the ``dim`` argument
        topk: number of the highest entries to turn into 1s
        dim: dimension on which to compare entries

    Returns:
        A binary tensor of the same shape as the input tensor of type ``torch.int32``

    Example:
        >>> x = torch.tensor([[1.1, 2.0, 3.0], [2.0, 1.0, 0.5]])
        >>> select_topk(x, topk=2)
        tensor([[0, 1, 1],
                [1, 1, 0]], dtype=torch.int32)

    """
    topk_tensor = torch.zeros_like(prob_tensor, dtype=torch.int)
    if topk == 1:
        topk_tensor.scatter_(dim, prob_tensor.argmax(dim=dim, keepdim=True), 1.0)
    else:
        topk_tensor.scatter_(dim, _top_k_with_half_precision_support(prob_tensor, k=topk, dim=dim), 1.0)
    return topk_tensor.int()