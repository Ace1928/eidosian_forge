from typing import Any, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCH_FIDELITY_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def maximum_mean_discrepancy(k_xx: Tensor, k_xy: Tensor, k_yy: Tensor) -> Tensor:
    """Adapted from `KID Score`_."""
    m = k_xx.shape[0]
    diag_x = torch.diag(k_xx)
    diag_y = torch.diag(k_yy)
    kt_xx_sums = k_xx.sum(dim=-1) - diag_x
    kt_yy_sums = k_yy.sum(dim=-1) - diag_y
    k_xy_sums = k_xy.sum(dim=0)
    kt_xx_sum = kt_xx_sums.sum()
    kt_yy_sum = kt_yy_sums.sum()
    k_xy_sum = k_xy_sums.sum()
    value = (kt_xx_sum + kt_yy_sum) / (m * (m - 1))
    value -= 2 * k_xy_sum / m ** 2
    return value