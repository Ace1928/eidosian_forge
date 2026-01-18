import math
from typing import Tuple
import torch
from torch import Tensor, tensor
def peak_signal_noise_ratio_with_blocked_effect(preds: Tensor, target: Tensor, block_size: int=8) -> Tensor:
    """Computes `Peak Signal to Noise Ratio With Blocked Effect` (PSNRB) metrics.

    .. math::
        \\text{PSNRB}(I, J) = 10 * \\log_{10} \\left(\\frac{\\max(I)^2}{\\text{MSE}(I, J)-\\text{B}(I, J)}\\right)

    Where :math:`\\text{MSE}` denotes the `mean-squared-error`_ function.

    Args:
        preds: estimated signal
        target: groun truth signal
        block_size: integer indication the block size

    Return:
        Tensor with PSNRB score

    Example:
        >>> import torch
        >>> from torchmetrics.functional.image import peak_signal_noise_ratio_with_blocked_effect
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand(1, 1, 28, 28)
        >>> target = torch.rand(1, 1, 28, 28)
        >>> peak_signal_noise_ratio_with_blocked_effect(preds, target)
        tensor(7.8402)

    """
    data_range = target.max() - target.min()
    sum_squared_error, bef, num_obs = _psnrb_update(preds, target, block_size=block_size)
    return _psnrb_compute(sum_squared_error, bef, num_obs, data_range)