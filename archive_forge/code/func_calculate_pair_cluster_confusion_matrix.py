from typing import Optional, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
def calculate_pair_cluster_confusion_matrix(preds: Optional[Tensor]=None, target: Optional[Tensor]=None, contingency: Optional[Tensor]=None) -> Tensor:
    """Calculates the pair cluster confusion matrix.

    Can either be calculated from predicted cluster labels and target cluster labels or from a pre-computed
    contingency matrix. The pair cluster confusion matrix is a 2x2 matrix where that defines the similarity between
    two clustering by considering all pairs of samples and counting pairs that are assigned into same or different
    clusters in the predicted and target clusterings.

    Note that the matrix is not symmetric.

    Inspired by:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.pair_confusion_matrix.html

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels
        contingency: contingency matrix

    Returns:
        A 2x2 tensor containing the pair cluster confusion matrix.

    Raises:
        ValueError:
            If neither `preds` and `target` nor `contingency` are provided.
        ValueError:
            If both `preds` and `target` and `contingency` are provided.

    Example:
        >>> import torch
        >>> from torchmetrics.functional.clustering.utils import calculate_pair_cluster_confusion_matrix
        >>> preds = torch.tensor([0, 0, 1, 1])
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> calculate_pair_cluster_confusion_matrix(preds, target)
        tensor([[8, 0],
                [0, 4]])
        >>> preds = torch.tensor([0, 0, 1, 2])
        >>> target = torch.tensor([0, 0, 1, 1])
        >>> calculate_pair_cluster_confusion_matrix(preds, target)
        tensor([[8, 2],
                [0, 2]])

    """
    if preds is None and target is None and (contingency is None):
        raise ValueError('Must provide either `preds` and `target` or `contingency`.')
    if preds is not None and target is not None and (contingency is not None):
        raise ValueError('Must provide either `preds` and `target` or `contingency`, not both.')
    if preds is not None and target is not None:
        contingency = calculate_contingency_matrix(preds, target)
    if contingency is None:
        raise ValueError('Must provide `contingency` if `preds` and `target` are not provided.')
    num_samples = contingency.sum()
    sum_c = contingency.sum(dim=1)
    sum_k = contingency.sum(dim=0)
    sum_squared = (contingency ** 2).sum()
    pair_matrix = torch.zeros(2, 2, dtype=contingency.dtype, device=contingency.device)
    pair_matrix[1, 1] = sum_squared - num_samples
    pair_matrix[1, 0] = (contingency * sum_k).sum() - sum_squared
    pair_matrix[0, 1] = (contingency.T * sum_c).sum() - sum_squared
    pair_matrix[0, 0] = num_samples ** 2 - pair_matrix[0, 1] - pair_matrix[1, 0] - sum_squared
    return pair_matrix