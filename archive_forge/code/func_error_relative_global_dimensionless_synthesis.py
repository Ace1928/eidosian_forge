from typing import Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
def error_relative_global_dimensionless_synthesis(preds: Tensor, target: Tensor, ratio: float=4, reduction: Literal['elementwise_mean', 'sum', 'none', None]='elementwise_mean') -> Tensor:
    """Calculates `Error relative global dimensionless synthesis`_ (ERGAS) metric.

    Args:
        preds: estimated image
        target: ground truth image
        ratio: ratio of high resolution to low resolution
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Return:
        Tensor with RelativeG score

    Raises:
        TypeError:
            If ``preds`` and ``target`` don't have the same data type.
        ValueError:
            If ``preds`` and ``target`` don't have ``BxCxHxW shape``.

    Example:
        >>> from torchmetrics.functional.image import error_relative_global_dimensionless_synthesis
        >>> gen = torch.manual_seed(42)
        >>> preds = torch.rand([16, 1, 16, 16], generator=gen)
        >>> target = preds * 0.75
        >>> ergds = error_relative_global_dimensionless_synthesis(preds, target)
        >>> torch.round(ergds)
        tensor(154.)

    """
    preds, target = _ergas_update(preds, target)
    return _ergas_compute(preds, target, ratio, reduction)