from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
from torch import Tensor
from torchaudio._extension import fail_if_no_align
@fail_if_no_align
def forced_align(log_probs: Tensor, targets: Tensor, input_lengths: Optional[Tensor]=None, target_lengths: Optional[Tensor]=None, blank: int=0) -> Tuple[Tensor, Tensor]:
    """Align a CTC label sequence to an emission.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Args:
        log_probs (Tensor): log probability of CTC emission output.
            Tensor of shape `(B, T, C)`. where `B` is the batch size, `T` is the input length,
            `C` is the number of characters in alphabet including blank.
        targets (Tensor): Target sequence. Tensor of shape `(B, L)`,
            where `L` is the target length.
        input_lengths (Tensor or None, optional):
            Lengths of the inputs (max value must each be <= `T`). 1-D Tensor of shape `(B,)`.
        target_lengths (Tensor or None, optional):
            Lengths of the targets. 1-D Tensor of shape `(B,)`.
        blank_id (int, optional): The index of blank symbol in CTC emission. (Default: 0)

    Returns:
        Tuple(Tensor, Tensor):
            Tensor: Label for each time step in the alignment path computed using forced alignment.

            Tensor: Log probability scores of the labels for each time step.

    Note:
        The sequence length of `log_probs` must satisfy:


        .. math::
            L_{\\text{log\\_probs}} \\ge L_{\\text{label}} + N_{\\text{repeat}}

        where :math:`N_{\\text{repeat}}` is the number of consecutively repeated tokens.
        For example, in str `"aabbc"`, the number of repeats are `2`.

    Note:
        The current version only supports ``batch_size==1``.
    """
    if blank in targets:
        raise ValueError(f"targets Tensor shouldn't contain blank index. Found {targets}.")
    if torch.max(targets) >= log_probs.shape[-1]:
        raise ValueError('targets values must be less than the CTC dimension')
    if input_lengths is None:
        batch_size, length = (log_probs.size(0), log_probs.size(1))
        input_lengths = torch.full((batch_size,), length, dtype=torch.int64, device=log_probs.device)
    if target_lengths is None:
        batch_size, length = (targets.size(0), targets.size(1))
        target_lengths = torch.full((batch_size,), length, dtype=torch.int64, device=targets.device)
    assert input_lengths is not None
    assert target_lengths is not None
    paths, scores = torch.ops.torchaudio.forced_align(log_probs, targets, input_lengths, target_lengths, blank)
    return (paths, scores)