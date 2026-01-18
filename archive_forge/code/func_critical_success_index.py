from typing import Optional, Tuple
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide
def critical_success_index(preds: Tensor, target: Tensor, threshold: float, keep_sequence_dim: Optional[int]=None) -> Tensor:
    """Compute critical success index.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        threshold: Values above or equal to threshold are replaced with 1, below by 0
        keep_sequence_dim: Index of the sequence dimension if the inputs are sequences of images. If specified,
            the score will be calculated separately for each image in the sequence. If ``None``, the score will be
            calculated across all dimensions.

    Returns:
        If ``keep_sequence_dim`` is specified, the metric returns a vector of  with CSI scores for each image
        in the sequence. Otherwise, it returns a scalar tensor with the CSI score.

    Example:
        >>> import torch
        >>> from torchmetrics.functional.regression import critical_success_index
        >>> x = torch.Tensor([[0.2, 0.7], [0.9, 0.3]])
        >>> y = torch.Tensor([[0.4, 0.2], [0.8, 0.6]])
        >>> critical_success_index(x, y, 0.5)
        tensor(0.3333)

    Example:
        >>> import torch
        >>> from torchmetrics.functional.regression import critical_success_index
        >>> x = torch.Tensor([[[0.2, 0.7], [0.9, 0.3]], [[0.2, 0.7], [0.9, 0.3]]])
        >>> y = torch.Tensor([[[0.4, 0.2], [0.8, 0.6]], [[0.4, 0.2], [0.8, 0.6]]])
        >>> critical_success_index(x, y, 0.5, keep_sequence_dim=0)
        tensor([0.3333, 0.3333])

    """
    hits, misses, false_alarms = _critical_success_index_update(preds, target, threshold, keep_sequence_dim)
    return _critical_success_index_compute(hits, misses, false_alarms)