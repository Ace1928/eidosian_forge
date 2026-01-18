import torch
from torch import Tensor
from torchmetrics.functional.regression.pearson import _pearson_corrcoef_compute, _pearson_corrcoef_update
def concordance_corrcoef(preds: Tensor, target: Tensor) -> Tensor:
    """Compute concordance correlation coefficient that measures the agreement between two variables.

    .. math::
        \\rho_c = \\frac{2 \\rho \\sigma_x \\sigma_y}{\\sigma_x^2 + \\sigma_y^2 + (\\mu_x - \\mu_y)^2}

    where :math:`\\mu_x, \\mu_y` is the means for the two variables, :math:`\\sigma_x^2, \\sigma_y^2` are the corresponding
    variances and \\rho is the pearson correlation coefficient between the two variables.

    Args:
        preds: estimated scores
        target: ground truth scores

    Example (single output regression):
        >>> from torchmetrics.functional.regression import concordance_corrcoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> concordance_corrcoef(preds, target)
        tensor([0.9777])

    Example (multi output regression):
        >>> from torchmetrics.functional.regression import concordance_corrcoef
        >>> target = torch.tensor([[3, -0.5], [2, 7]])
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> concordance_corrcoef(preds, target)
        tensor([0.7273, 0.9887])

    """
    d = preds.shape[1] if preds.ndim == 2 else 1
    _temp = torch.zeros(d, dtype=preds.dtype, device=preds.device)
    mean_x, mean_y, var_x = (_temp.clone(), _temp.clone(), _temp.clone())
    var_y, corr_xy, nb = (_temp.clone(), _temp.clone(), _temp.clone())
    mean_x, mean_y, var_x, var_y, corr_xy, nb = _pearson_corrcoef_update(preds, target, mean_x, mean_y, var_x, var_y, corr_xy, nb, num_outputs=1 if preds.ndim == 1 else preds.shape[-1])
    return _concordance_corrcoef_compute(mean_x, mean_y, var_x, var_y, corr_xy, nb)