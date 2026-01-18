from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.functional.image.utils import _uniform_filter
from torchmetrics.utilities.checks import _check_same_shape
def root_mean_squared_error_using_sliding_window(preds: Tensor, target: Tensor, window_size: int=8, return_rmse_map: bool=False) -> Union[Optional[Tensor], Tuple[Optional[Tensor], Tensor]]:
    """Compute Root Mean Squared Error (RMSE) using sliding window.

    Args:
        preds: Deformed image
        target: Ground truth image
        window_size: Sliding window used for rmse calculation
        return_rmse_map: An indication whether the full rmse reduced image should be returned.

    Return:
        RMSE using sliding window
        (Optionally) RMSE map

    Example:
        >>> from torchmetrics.functional.image import root_mean_squared_error_using_sliding_window
        >>> g = torch.manual_seed(22)
        >>> preds = torch.rand(4, 3, 16, 16)
        >>> target = torch.rand(4, 3, 16, 16)
        >>> root_mean_squared_error_using_sliding_window(preds, target)
        tensor(0.3999)

    Raises:
        ValueError: If ``window_size`` is not a positive integer.

    """
    if not isinstance(window_size, int) or (isinstance(window_size, int) and window_size < 1):
        raise ValueError('Argument `window_size` is expected to be a positive integer.')
    rmse_val_sum, rmse_map, total_images = _rmse_sw_update(preds, target, window_size, rmse_val_sum=None, rmse_map=None, total_images=None)
    rmse, rmse_map = _rmse_sw_compute(rmse_val_sum, rmse_map, total_images)
    if return_rmse_map:
        return (rmse, rmse_map)
    return rmse