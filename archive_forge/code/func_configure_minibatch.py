import itertools
from functools import partial
from typing import (
from ..util import minibatch, registry
@registry.batchers('spacy.batch_by_sequence.v1')
def configure_minibatch(size: Sizing, get_length: Optional[Callable[[ItemT], int]]=None) -> BatcherT:
    """Create a batcher that creates batches of the specified size.

    size (int or Sequence[int]): The target number of items per batch.
        Can be a single integer, or a sequence, allowing for variable batch sizes.
    """
    optionals = {'get_length': get_length} if get_length is not None else {}
    return partial(minibatch, size=size, **optionals)