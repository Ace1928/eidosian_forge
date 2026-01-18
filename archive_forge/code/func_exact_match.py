from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.enums import ClassificationTaskNoBinary
def exact_match(preds: Tensor, target: Tensor, task: Literal['multiclass', 'multilabel'], num_classes: Optional[int]=None, num_labels: Optional[int]=None, threshold: float=0.5, multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """Compute Exact match (also known as subset accuracy).

    Exact Match is a stricter version of accuracy where all classes/labels have to match exactly for the sample to be
    correctly classified.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :func:`~torchmetrics.functional.classification.multiclass_exact_match` and
    :func:`~torchmetrics.functional.classification.multilabel_exact_match` for the specific details of
    each argument influence and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 1], [2, 1], [0, 2]], [[2, 2], [2, 1], [1, 0]]])
        >>> exact_match(preds, target, task="multiclass", num_classes=3, multidim_average='global')
        tensor(0.5000)

        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 1], [2, 1], [0, 2]], [[2, 2], [2, 1], [1, 0]]])
        >>> exact_match(preds, target, task="multiclass", num_classes=3, multidim_average='samplewise')
        tensor([1., 0.])

    """
    task = ClassificationTaskNoBinary.from_str(task)
    if task == ClassificationTaskNoBinary.MULTICLASS:
        assert num_classes is not None
        return multiclass_exact_match(preds, target, num_classes, multidim_average, ignore_index, validate_args)
    if task == ClassificationTaskNoBinary.MULTILABEL:
        assert num_labels is not None
        return multilabel_exact_match(preds, target, num_labels, threshold, multidim_average, ignore_index, validate_args)
    raise ValueError(f'Not handled value: {task}')