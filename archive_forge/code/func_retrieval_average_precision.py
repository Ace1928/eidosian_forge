from typing import Optional
import torch
from torch import Tensor, tensor
from torchmetrics.utilities.checks import _check_retrieval_functional_inputs
def retrieval_average_precision(preds: Tensor, target: Tensor, top_k: Optional[int]=None) -> Tensor:
    """Compute average precision (for information retrieval), as explained in `IR Average precision`_.

    ``preds`` and ``target`` should be of the same shape and live on the same device. If no ``target`` is ``True``,
    ``0`` is returned. ``target`` must be either `bool` or `integers` and ``preds`` must be ``float``,
    otherwise an error is raised.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant or not.
        top_k: consider only the top k elements (default: ``None``, which considers them all)

    Return:
        a single-value tensor with the average precision (AP) of the predictions ``preds`` w.r.t. the labels ``target``.

    Raises:
        ValueError:
            If ``top_k`` is not ``None`` or an integer larger than 0.

    Example:
        >>> from torchmetrics.functional.retrieval import retrieval_average_precision
        >>> preds = tensor([0.2, 0.3, 0.5])
        >>> target = tensor([True, False, True])
        >>> retrieval_average_precision(preds, target)
        tensor(0.8333)

    """
    preds, target = _check_retrieval_functional_inputs(preds, target)
    top_k = top_k or preds.shape[-1]
    if not isinstance(top_k, int) and top_k <= 0:
        raise ValueError(f'Argument ``top_k`` has to be a positive integer or None, but got {top_k}.')
    target = target[preds.topk(min(top_k, preds.shape[-1]), sorted=True, dim=-1)[1]]
    if not target.sum():
        return tensor(0.0, device=preds.device)
    positions = torch.arange(1, len(target) + 1, device=target.device, dtype=torch.float32)[target > 0]
    return torch.div(torch.arange(len(positions), device=positions.device, dtype=torch.float32) + 1, positions).mean()