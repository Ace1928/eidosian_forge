from typing import Tuple
import torch
from torch._C import DispatchKey, DispatchKeySet
from torch._prims_common import is_expandable_to
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from torch.utils.weak import WeakTensorKeyDictionary
from typing import *  # noqa: F403
def jagged_from_tensor_and_lengths(tensor: torch.Tensor, starts: torch.Tensor, lengths: torch.Tensor) -> Tuple[NestedTensor, torch.Tensor, Optional[torch.Tensor]]:
    """Constructs a NestedTensor backed by jagged layout from a tensor, starts of sequences, and sequence lengths"""
    batch_size = tensor.shape[0]
    if is_expandable_to(starts.shape, (batch_size,)) and is_expandable_to(lengths.shape, (batch_size,)):
        start_list = starts.expand(batch_size)
        length_list = lengths.expand(batch_size)
    else:
        raise RuntimeError('When constructing a jagged nested tensor using narrow(), your start and length must be Tensors that broadcast to input.shape[0]')
    assert len(tensor.shape) >= 2, 'tensor must at least be 2D for the nested narrow op to work'
    max_seq_len = tensor.shape[1]
    offset_lengths = max_seq_len * torch.arange(0, batch_size, dtype=torch.int64, device=tensor.device)
    offsets = torch.cat([start_list + offset_lengths, (start_list[-1] + offset_lengths[-1] + length_list[-1]).unsqueeze(0)])
    if len(tensor.shape) > 2:
        values = tensor.view(-1, *tensor.shape[2:])
    else:
        values = tensor.view(-1)
    is_contiguous = True
    orig_dim = tensor.shape[1]
    if torch.any(length_list[1:-1].ne(orig_dim)):
        is_contiguous = False
    if torch.any(offsets[1:-2].diff().ne(orig_dim)):
        is_contiguous = False
    if offsets[0] + length_list[0] != orig_dim:
        is_contiguous = False
    actual_max_seqlen = int(torch.max(lengths).item())
    min_seqlen = int(torch.min(lengths).item())
    if is_contiguous:
        return (ViewNestedFromBuffer.apply(values[offsets[0]:offsets[-1]], offsets - offsets[0], actual_max_seqlen, min_seqlen), offsets, None)
    return (ViewNonContiguousNestedFromBuffer.apply(values, offsets, length_list, actual_max_seqlen, min_seqlen), offsets, length_list)