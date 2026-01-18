import itertools
from functools import partial
from typing import (
from ..util import minibatch, registry
def minibatch_by_padded_size(seqs: Iterable[ItemT], size: Sizing, buffer: int=256, discard_oversize: bool=False, get_length: Callable=len) -> Iterable[List[ItemT]]:
    """Minibatch a sequence by the size of padded batches that would result,
    with sequences binned by length within a window.

    The padded size is defined as the maximum length of sequences within the
    batch multiplied by the number of sequences in the batch.

    size (int or Sequence[int]): The largest padded size to batch sequences into.
    buffer (int): The number of sequences to accumulate before sorting by length.
        A larger buffer will result in more even sizing, but if the buffer is
        very large, the iteration order will be less random, which can result
        in suboptimal training.
    discard_oversize (bool): Whether to discard sequences that are by themselves
        longer than the largest padded batch size.
    get_length (Callable or None): Function to get the length of a sequence item.
        The `len` function is used by default.
    """
    if isinstance(size, int):
        size_ = itertools.repeat(size)
    else:
        size_ = iter(size)
    for outer_batch in minibatch(seqs, size=buffer):
        outer_batch = list(outer_batch)
        target_size = next(size_)
        for indices in _batch_by_length(outer_batch, target_size, get_length):
            subbatch = [outer_batch[i] for i in indices]
            padded_size = max((len(seq) for seq in subbatch)) * len(subbatch)
            if discard_oversize and padded_size >= target_size:
                pass
            else:
                yield subbatch