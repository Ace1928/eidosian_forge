import functools
import math
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import conv2d, conv3d, pad, unfold
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE
def distance_transform(x: Tensor, sampling: Optional[Union[Tensor, List[float]]]=None, metric: Literal['euclidean', 'chessboard', 'taxicab']='euclidean', engine: Literal['pytorch', 'scipy']='pytorch') -> Tensor:
    """Calculate distance transform of a binary tensor.

    This function calculates the distance transform of a binary tensor, replacing each foreground pixel with the
    distance to the closest background pixel. The distance is calculated using the euclidean, chessboard or taxicab
    distance.

    The memory consumption of this function is in the worst cast N/2**2 where N is the number of pixel. Since we need
    to compare all foreground pixels to all background pixels, the memory consumption is quadratic in the number of
    pixels. The memory consumption can be reduced by using the ``scipy`` engine, which is more memory efficient but
    should also be slower for larger images.

    Args:
        x: The binary tensor to calculate the distance transform of.
        sampling: Only relevant when distance is calculated using the euclidean distance. The sampling refers to the
            pixel spacing in the image, i.e. the distance between two adjacent pixels. If not provided, the pixel
            spacing is assumed to be 1.
        metric: The distance to use for the distance transform. Can be one of ``"euclidean"``, ``"chessboard"``
            or ``"taxicab"``.
        engine: The engine to use for the distance transform. Can be one of ``["pytorch", "scipy"]``. In general,
            the ``pytorch`` engine is faster, but the ``scipy`` engine is more memory efficient.

    Returns:
        The distance transform of the input tensor.

    Examples::
        >>> from torchmetrics.functional.segmentation.utils import distance_transform
        >>> import torch
        >>> x = torch.tensor([[0, 0, 0, 0, 0],
        ...                   [0, 1, 1, 1, 0],
        ...                   [0, 1, 1, 1, 0],
        ...                   [0, 1, 1, 1, 0],
        ...                   [0, 0, 0, 0, 0]])
        >>> distance_transform(x)
        tensor([[0., 0., 0., 0., 0.],
                [0., 1., 1., 1., 0.],
                [0., 1., 2., 1., 0.],
                [0., 1., 1., 1., 0.],
                [0., 0., 0., 0., 0.]])

    """
    if not isinstance(x, Tensor):
        raise ValueError(f'Expected argument `x` to be of type `torch.Tensor` but got `{type(x)}`.')
    if x.ndim != 2:
        raise ValueError(f'Expected argument `x` to be of rank 2 but got rank `{x.ndim}`.')
    if sampling is not None and (not isinstance(sampling, list)):
        raise ValueError(f'Expected argument `sampling` to either be `None` or of type `list` but got `{type(sampling)}`.')
    if metric not in ['euclidean', 'chessboard', 'taxicab']:
        raise ValueError(f"Expected argument `metric` to be one of `['euclidean', 'chessboard', 'taxicab']` but got `{metric}`.")
    if engine not in ['pytorch', 'scipy']:
        raise ValueError(f"Expected argument `engine` to be one of `['pytorch', 'scipy']` but got `{engine}`.")
    if sampling is None:
        sampling = [1, 1]
    elif len(sampling) != 2:
        raise ValueError(f'Expected argument `sampling` to have length 2 but got length `{len(sampling)}`.')
    if engine == 'pytorch':
        i0, j0 = torch.where(x == 0)
        i1, j1 = torch.where(x == 1)
        dis_row = (i1.unsqueeze(1) - i0.unsqueeze(0)).abs_().mul_(sampling[0])
        dis_col = (j1.unsqueeze(1) - j0.unsqueeze(0)).abs_().mul_(sampling[1])
        h, _ = x.shape
        if metric == 'euclidean':
            dis_row = dis_row.float()
            dis_row.pow_(2).add_(dis_col.pow_(2)).sqrt_()
        if metric == 'chessboard':
            dis_row = dis_row.max(dis_col)
        if metric == 'taxicab':
            dis_row.add_(dis_col)
        mindis, _ = torch.min(dis_row, dim=1)
        z = torch.zeros_like(x, dtype=mindis.dtype).view(-1)
        z[i1 * h + j1] = mindis
        return z.view(x.shape)
    if not _SCIPY_AVAILABLE:
        raise ValueError('The `scipy` engine requires `scipy` to be installed. Either install `scipy` or use the `pytorch` engine.')
    from scipy import ndimage
    if metric == 'euclidean':
        return ndimage.distance_transform_edt(x.cpu().numpy(), sampling)
    return ndimage.distance_transform_cdt(x.cpu().numpy(), metric=metric)