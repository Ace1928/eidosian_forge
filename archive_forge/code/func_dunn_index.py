from itertools import combinations
from typing import Tuple
import torch
from torch import Tensor
def dunn_index(data: Tensor, labels: Tensor, p: float=2) -> Tensor:
    """Compute the Dunn index.

    Args:
        data: feature vectors
        labels: cluster labels
        p: p-norm used for distance metric

    Returns:
        scalar tensor with the dunn index

    Example:
        >>> from torchmetrics.functional.clustering import dunn_index
        >>> data = torch.tensor([[0, 0], [0.5, 0], [1, 0], [0.5, 1]])
        >>> labels = torch.tensor([0, 0, 0, 1])
        >>> dunn_index(data, labels)
        tensor(2.)

    """
    pairwise_distance, max_distance = _dunn_index_update(data, labels, p)
    return _dunn_index_compute(pairwise_distance, max_distance)