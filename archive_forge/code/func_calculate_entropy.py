from typing import Optional, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
def calculate_entropy(x: Tensor) -> Tensor:
    """Calculate entropy for a tensor of labels.

    Final calculation of entropy is performed in log form to account for roundoff error.

    Args:
        x: labels

    Returns:
        entropy: entropy of tensor

    Example:
        >>> from torchmetrics.functional.clustering.utils import calculate_entropy
        >>> labels = torch.tensor([1, 3, 2, 2, 1])
        >>> calculate_entropy(labels)
        tensor(1.0549)

    """
    if len(x) == 0:
        return tensor(1.0, device=x.device)
    p = torch.bincount(torch.unique(x, return_inverse=True)[1])
    p = p[p > 0]
    if p.size() == 1:
        return tensor(0.0, device=x.device)
    n = p.sum()
    return -torch.sum(p / n * (torch.log(p) - torch.log(n)))