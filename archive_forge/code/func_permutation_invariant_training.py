from itertools import permutations
from typing import Any, Callable, Tuple
import numpy as np
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE
def permutation_invariant_training(preds: Tensor, target: Tensor, metric_func: Callable, mode: Literal['speaker-wise', 'permutation-wise']='speaker-wise', eval_func: Literal['max', 'min']='max', **kwargs: Any) -> Tuple[Tensor, Tensor]:
    """Calculate `Permutation invariant training`_ (PIT).

    This metric can evaluate models for speaker independent multi-talker speech separation in a permutation
    invariant way.

    Args:
        preds: float tensor with shape ``(batch_size,num_speakers,...)``
        target: float tensor with shape ``(batch_size,num_speakers,...)``
        metric_func: a metric function accept a batch of target and estimate.
            if `mode`==`'speaker-wise'`, then ``metric_func(preds[:, i, ...], target[:, j, ...])`` is called
            and expected to return a batch of metric tensors ``(batch,)``;

            if `mode`==`'permutation-wise'`, then ``metric_func(preds[:, p, ...], target[:, :, ...])`` is called,
            where `p` is one possible permutation, e.g. [0,1] or [1,0] for 2-speaker case, and expected to return
            a batch of metric tensors ``(batch,)``;

        mode: can be `'speaker-wise'` or `'permutation-wise'`.
        eval_func: the function to find the best permutation, can be ``'min'`` or ``'max'``,
            i.e. the smaller the better or the larger the better.
        kwargs: Additional args for metric_func

    Returns:
        Tuple of two float tensors. First tensor with shape ``(batch,)`` contains the best metric value for each sample
        and second tensor with shape ``(batch,)`` contains the best permutation.

    Example:
        >>> from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
        >>> # [batch, spk, time]
        >>> preds = torch.tensor([[[-0.0579,  0.3560, -0.9604], [-0.1719,  0.3205,  0.2951]]])
        >>> target = torch.tensor([[[ 1.0958, -0.1648,  0.5228], [-0.4100,  1.1942, -0.5103]]])
        >>> best_metric, best_perm = permutation_invariant_training(
        ...     preds, target, scale_invariant_signal_distortion_ratio,
        ...     mode="speaker-wise", eval_func="max")
        >>> best_metric
        tensor([-5.1091])
        >>> best_perm
        tensor([[0, 1]])
        >>> pit_permutate(preds, best_perm)
        tensor([[[-0.0579,  0.3560, -0.9604],
                 [-0.1719,  0.3205,  0.2951]]])

    """
    if preds.shape[0:2] != target.shape[0:2]:
        raise RuntimeError('Predictions and targets are expected to have the same shape at the batch and speaker dimensions')
    if eval_func not in ['max', 'min']:
        raise ValueError(f'eval_func can only be "max" or "min" but got {eval_func}')
    if mode not in ['speaker-wise', 'permutation-wise']:
        raise ValueError(f'mode can only be "speaker-wise" or "permutation-wise" but got {eval_func}')
    if target.ndim < 2:
        raise ValueError(f'Inputs must be of shape [batch, spk, ...], got {target.shape} and {preds.shape} instead')
    eval_op = torch.max if eval_func == 'max' else torch.min
    batch_size, spk_num = target.shape[0:2]
    if mode == 'permutation-wise':
        perms = _gen_permutations(spk_num=spk_num, device=preds.device)
        perm_num = perms.shape[0]
        ppreds = torch.index_select(preds, dim=1, index=perms.reshape(-1)).reshape(batch_size * perm_num, *preds.shape[1:])
        ptarget = target.repeat_interleave(repeats=perm_num, dim=0)
        metric_of_ps = metric_func(ppreds, ptarget)
        metric_of_ps = torch.mean(metric_of_ps.reshape(batch_size, len(perms), -1), dim=-1)
        best_metric, best_indexes = eval_op(metric_of_ps, dim=1)
        best_indexes = best_indexes.detach()
        best_perm = perms[best_indexes, :]
        return (best_metric, best_perm)
    first_ele = metric_func(preds[:, 0, ...], target[:, 0, ...], **kwargs)
    metric_mtx = torch.empty((batch_size, spk_num, spk_num), dtype=first_ele.dtype, device=first_ele.device)
    metric_mtx[:, 0, 0] = first_ele
    for target_idx in range(spk_num):
        for preds_idx in range(spk_num):
            if target_idx == 0 and preds_idx == 0:
                continue
            metric_mtx[:, target_idx, preds_idx] = metric_func(preds[:, preds_idx, ...], target[:, target_idx, ...], **kwargs)
    if spk_num < 3 or not _SCIPY_AVAILABLE:
        if spk_num >= 3 and (not _SCIPY_AVAILABLE):
            rank_zero_warn(f'In pit metric for speaker-num {spk_num}>3, we recommend installing scipy for better performance')
        best_metric, best_perm = _find_best_perm_by_exhaustive_method(metric_mtx, eval_op)
    else:
        best_metric, best_perm = _find_best_perm_by_linear_sum_assignment(metric_mtx, eval_op)
    return (best_metric, best_perm)