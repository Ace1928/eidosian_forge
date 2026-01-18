import functools
import math
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import conv2d, conv3d, pad, unfold
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE
def surface_distance(preds: Tensor, target: Tensor, distance_metric: Literal['euclidean', 'chessboard', 'taxicab']='euclidean', spacing: Optional[Union[Tensor, List[float]]]=None) -> Tensor:
    """Calculate the surface distance between two binary edge masks.

    May return infinity if the predicted mask is empty and the target mask is not, or vice versa.

    Args:
        preds: The predicted binary edge mask.
        target: The target binary edge mask.
        distance_metric: The distance metric to use. One of `["euclidean", "chessboard", "taxicab"]`.
        spacing: The spacing between pixels along each spatial dimension.

    Returns:
        A tensor with length equal to the number of edges in predictions e.g. `preds.sum()`. Each element is the
        distance from the corresponding edge in `preds` to the closest edge in `target`.

    Example::
        >>> import torch
        >>> from torchmetrics.functional.segmentation.utils import surface_distance
        >>> preds = torch.tensor([[1, 1, 1, 1, 1],
        ...                       [1, 0, 0, 0, 1],
        ...                       [1, 0, 0, 0, 1],
        ...                       [1, 0, 0, 0, 1],
        ...                       [1, 1, 1, 1, 1]], dtype=torch.bool)
        >>> target = torch.tensor([[1, 1, 1, 1, 0],
        ...                        [1, 0, 0, 1, 0],
        ...                        [1, 0, 0, 1, 0],
        ...                        [1, 0, 0, 1, 0],
        ...                        [1, 1, 1, 1, 0]], dtype=torch.bool)
        >>> surface_distance(preds, target, distance_metric="euclidean", spacing=[1, 1])
        tensor([0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 1.])

    """
    if not (preds.dtype == torch.bool and target.dtype == torch.bool):
        raise ValueError(f'Expected both inputs to be of type `torch.bool`, but got {preds.dtype} and {target.dtype}.')
    if not torch.any(target):
        dis = torch.inf * torch.ones_like(target)
    else:
        if not torch.any(preds):
            dis = torch.inf * torch.ones_like(preds)
            return dis[target]
        dis = distance_transform(~target, sampling=spacing, metric=distance_metric)
    return dis[preds]