from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.pairwise.helpers import _check_input, _reduce_distance_matrix
from torchmetrics.utilities.exceptions import TorchMetricsUserError
def pairwise_minkowski_distance(x: Tensor, y: Optional[Tensor]=None, exponent: float=2, reduction: Literal['mean', 'sum', 'none', None]=None, zero_diagonal: Optional[bool]=None) -> Tensor:
    """Calculate pairwise minkowski distances.

    .. math::
        d_{minkowski}(x,y,p) = ||x - y||_p = \\sqrt[p]{\\sum_{d=1}^D (x_d - y_d)^p}

    If both :math:`x` and :math:`y` are passed in, the calculation will be performed pairwise between the rows of
    :math:`x` and :math:`y`. If only :math:`x` is passed in, the calculation will be performed between the rows
    of :math:`x`.

    Args:
        x: Tensor with shape ``[N, d]``
        y: Tensor with shape ``[M, d]``, optional
        exponent: int or float larger than 1, exponent to which the difference between preds and target is to be raised
        reduction: reduction to apply along the last dimension. Choose between `'mean'`, `'sum'`
            (applied along column dimension) or  `'none'`, `None` for no reduction
        zero_diagonal: if the diagonal of the distance matrix should be set to 0. If only `x` is given
            this defaults to `True` else if `y` is also given it defaults to `False`

    Returns:
        A ``[N,N]`` matrix of distances if only ``x`` is given, else a ``[N,M]`` matrix

    Example:
        >>> import torch
        >>> from torchmetrics.functional.pairwise import pairwise_minkowski_distance
        >>> x = torch.tensor([[2, 3], [3, 5], [5, 8]], dtype=torch.float32)
        >>> y = torch.tensor([[1, 0], [2, 1]], dtype=torch.float32)
        >>> pairwise_minkowski_distance(x, y, exponent=4)
        tensor([[3.0092, 2.0000],
                [5.0317, 4.0039],
                [8.1222, 7.0583]])
        >>> pairwise_minkowski_distance(x, exponent=4)
        tensor([[0.0000, 2.0305, 5.1547],
                [2.0305, 0.0000, 3.1383],
                [5.1547, 3.1383, 0.0000]])

    """
    distance = _pairwise_minkowski_distance_update(x, y, exponent, zero_diagonal)
    return _reduce_distance_matrix(distance, reduction)