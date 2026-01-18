import itertools
from functools import partial
from typing import (
from ..util import minibatch, registry
@registry.batchers('spacy.batch_by_padded.v1')
def configure_minibatch_by_padded_size(*, size: Sizing, buffer: int, discard_oversize: bool, get_length: Optional[Callable[[ItemT], int]]=None) -> BatcherT:
    """Create a batcher that uses the `batch_by_padded_size` strategy.

    The padded size is defined as the maximum length of sequences within the
    batch multiplied by the number of sequences in the batch.

    size (int or Sequence[int]): The largest padded size to batch sequences into.
        Can be a single integer, or a sequence, allowing for variable batch sizes.
    buffer (int): The number of sequences to accumulate before sorting by length.
        A larger buffer will result in more even sizing, but if the buffer is
        very large, the iteration order will be less random, which can result
        in suboptimal training.
    discard_oversize (bool): Whether to discard sequences that are by themselves
        longer than the largest padded batch size.
    get_length (Callable or None): Function to get the length of a sequence item.
        The `len` function is used by default.
    """
    optionals = {'get_length': get_length} if get_length is not None else {}
    return partial(minibatch_by_padded_size, size=size, buffer=buffer, discard_oversize=discard_oversize, **optionals)