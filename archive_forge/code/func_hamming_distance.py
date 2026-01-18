from typing import Optional
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
def hamming_distance(preds: Tensor, target: Tensor, task: Literal['binary', 'multiclass', 'multilabel'], threshold: float=0.5, num_classes: Optional[int]=None, num_labels: Optional[int]=None, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='micro', multidim_average: Optional[Literal['global', 'samplewise']]='global', top_k: Optional[int]=1, ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """Compute the average `Hamming distance`_ (also known as Hamming loss).

    .. math::
        \\text{Hamming distance} = \\frac{1}{N \\cdot L} \\sum_i^N \\sum_l^L 1(y_{il} \\neq \\hat{y}_{il})

    Where :math:`y` is a tensor of target values, :math:`\\hat{y}` is a tensor of predictions,
    and :math:`\\bullet_{il}` refers to the :math:`l`-th label of the :math:`i`-th sample of that
    tensor.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_hamming_distance`,
    :func:`~torchmetrics.functional.classification.multiclass_hamming_distance` and
    :func:`~torchmetrics.functional.classification.multilabel_hamming_distance` for
    the specific details of each argument influence and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> target = tensor([[0, 1], [1, 1]])
        >>> preds = tensor([[0, 1], [0, 1]])
        >>> hamming_distance(preds, target, task="binary")
        tensor(0.2500)

    """
    task = ClassificationTask.from_str(task)
    assert multidim_average is not None
    if task == ClassificationTask.BINARY:
        return binary_hamming_distance(preds, target, threshold, multidim_average, ignore_index, validate_args)
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
        if not isinstance(top_k, int):
            raise ValueError(f'`top_k` is expected to be `int` but `{type(top_k)} was passed.`')
        return multiclass_hamming_distance(preds, target, num_classes, average, top_k, multidim_average, ignore_index, validate_args)
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(f'`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`')
        return multilabel_hamming_distance(preds, target, num_labels, threshold, average, multidim_average, ignore_index, validate_args)
    raise ValueError(f'Not handled value: {task}')