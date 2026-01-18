import math
import tempfile
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
from torchaudio._internal.module_utils import deprecated
from .filtering import highpass_biquad, treble_biquad
def rnnt_loss(logits: Tensor, targets: Tensor, logit_lengths: Tensor, target_lengths: Tensor, blank: int=-1, clamp: float=-1, reduction: str='mean', fused_log_softmax: bool=True):
    """Compute the RNN Transducer loss from *Sequence Transduction with Recurrent Neural Networks*
    :cite:`graves2012sequence`.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    The RNN Transducer loss extends the CTC loss by defining a distribution over output
    sequences of all lengths, and by jointly modelling both input-output and output-output
    dependencies.

    Args:
        logits (Tensor): Tensor of dimension `(batch, max seq length, max target length + 1, class)`
            containing output from joiner
        targets (Tensor): Tensor of dimension `(batch, max target length)` containing targets with zero padded
        logit_lengths (Tensor): Tensor of dimension `(batch)` containing lengths of each sequence from encoder
        target_lengths (Tensor): Tensor of dimension `(batch)` containing lengths of targets for each sequence
        blank (int, optional): blank label (Default: ``-1``)
        clamp (float, optional): clamp for gradients (Default: ``-1``)
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. (Default: ``"mean"``)
        fused_log_softmax (bool): set to False if calling log_softmax outside of loss (Default: ``True``)
    Returns:
        Tensor: Loss with the reduction option applied. If ``reduction`` is  ``"none"``, then size `(batch)`,
        otherwise scalar.
    """
    if reduction not in ['none', 'mean', 'sum']:
        raise ValueError('reduction should be one of "none", "mean", or "sum"')
    if blank < 0:
        blank = logits.shape[-1] + blank
    costs, _ = torch.ops.torchaudio.rnnt_loss(logits=logits, targets=targets, logit_lengths=logit_lengths, target_lengths=target_lengths, blank=blank, clamp=clamp, fused_log_softmax=fused_log_softmax)
    if reduction == 'mean':
        return costs.mean()
    elif reduction == 'sum':
        return costs.sum()
    return costs