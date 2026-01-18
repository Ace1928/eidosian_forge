from typing import Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
def spectral_angle_mapper(preds: Tensor, target: Tensor, reduction: Literal['elementwise_mean', 'sum', 'none', None]='elementwise_mean') -> Tensor:
    """Universal Spectral Angle Mapper.

    Args:
        preds: estimated image
        target: ground truth image
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Return:
        Tensor with Spectral Angle Mapper score

    Raises:
        TypeError:
            If ``preds`` and ``target`` don't have the same data type.
        ValueError:
            If ``preds`` and ``target`` don't have ``BxCxHxW shape``.

    Example:
        >>> from torchmetrics.functional.image import spectral_angle_mapper
        >>> gen = torch.manual_seed(42)
        >>> preds = torch.rand([16, 3, 16, 16], generator=gen)
        >>> target = torch.rand([16, 3, 16, 16], generator=gen)
        >>> spectral_angle_mapper(preds, target)
        tensor(0.5914)

    References:
        [1] Roberta H. Yuhas, Alexander F. H. Goetz and Joe W. Boardman, "Discrimination among semi-arid
        landscape endmembers using the Spectral Angle Mapper (SAM) algorithm" in PL, Summaries of the Third Annual JPL
        Airborne Geoscience Workshop, vol. 1, June 1, 1992.

    """
    preds, target = _sam_update(preds, target)
    return _sam_compute(preds, target, reduction)