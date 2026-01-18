from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.utilities.distributed import reduce
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE
def spatial_distortion_index(preds: Tensor, ms: Tensor, pan: Tensor, pan_lr: Optional[Tensor]=None, norm_order: int=1, window_size: int=7, reduction: Literal['elementwise_mean', 'sum', 'none']='elementwise_mean') -> Tensor:
    """Calculate `Spatial Distortion Index`_ (SpatialDistortionIndex_) also known as D_s.

    Metric is used to compare the spatial distortion between two images.

    Args:
        preds: High resolution multispectral image.
        ms: Low resolution multispectral image.
        pan: High resolution panchromatic image.
        pan_lr: Low resolution panchromatic image.
        norm_order: Order of the norm applied on the difference.
        window_size: Window size of the filter applied to degrade the high resolution panchromatic image.
        reduction: A method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        Tensor with SpatialDistortionIndex score

    Raises:
        TypeError:
            If ``preds``, ``ms``, ``pan`` and ``pan_lr`` don't have the same data type.
        ValueError:
            If ``preds``, ``ms``, ``pan`` and ``pan_lr`` don't have ``BxCxHxW shape``.
        ValueError:
            If ``preds``, ``ms``, ``pan`` and ``pan_lr`` don't have the same batch and channel sizes.
        ValueError:
            If ``preds`` and ``pan`` don't have the same dimension.
        ValueError:
            If ``ms`` and ``pan_lr`` don't have the same dimension.
        ValueError:
            If ``preds`` and ``pan`` don't have dimension which is multiple of that of ``ms``.
        ValueError:
            If ``norm_order`` is not a positive integer.
        ValueError:
            If ``window_size`` is not a positive integer.

    Example:
        >>> from torchmetrics.functional.image import spatial_distortion_index
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand([16, 3, 32, 32])
        >>> ms = torch.rand([16, 3, 16, 16])
        >>> pan = torch.rand([16, 3, 32, 32])
        >>> spatial_distortion_index(preds, ms, pan)
        tensor(0.0090)

    """
    if not isinstance(norm_order, int) or norm_order <= 0:
        raise ValueError(f'Expected `norm_order` to be a positive integer. Got norm_order: {norm_order}.')
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError(f'Expected `window_size` to be a positive integer. Got window_size: {window_size}.')
    preds, ms, pan, pan_lr = _spatial_distortion_index_update(preds, ms, pan, pan_lr)
    return _spatial_distortion_index_compute(preds, ms, pan, pan_lr, norm_order, window_size, reduction)