import warnings
from typing import Iterable, List, NamedTuple, Tuple, Union
import torch
from torch import Tensor
from ... import _VF
from ..._jit_internal import Optional
def pack_sequence(sequences: List[Tensor], enforce_sorted: bool=True) -> PackedSequence:
    """Packs a list of variable length Tensors.

    Consecutive call of the next functions: ``pad_sequence``, ``pack_padded_sequence``.

    ``sequences`` should be a list of Tensors of size ``L x *``, where `L` is
    the length of a sequence and `*` is any number of trailing dimensions,
    including zero.

    For unsorted sequences, use `enforce_sorted = False`. If ``enforce_sorted``
    is ``True``, the sequences should be sorted in the order of decreasing length.
    ``enforce_sorted = True`` is only necessary for ONNX export.


    Example:
        >>> from torch.nn.utils.rnn import pack_sequence
        >>> a = torch.tensor([1, 2, 3])
        >>> b = torch.tensor([4, 5])
        >>> c = torch.tensor([6])
        >>> pack_sequence([a, b, c])
        PackedSequence(data=tensor([1, 4, 6, 2, 5, 3]), batch_sizes=tensor([3, 2, 1]), sorted_indices=None, unsorted_indices=None)


    Args:
        sequences (list[Tensor]): A list of sequences of decreasing length.
        enforce_sorted (bool, optional): if ``True``, checks that the input
            contains sequences sorted by length in a decreasing order. If
            ``False``, this condition is not checked. Default: ``True``.

    Returns:
        a :class:`PackedSequence` object
    """
    lengths = torch.as_tensor([v.size(0) for v in sequences])
    return pack_padded_sequence(pad_sequence(sequences), lengths, enforce_sorted=enforce_sorted)