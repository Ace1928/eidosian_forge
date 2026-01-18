from collections import deque
from itertools import islice
from typing import (
from typing_extensions import Literal
def batch_iterate(size: Optional[int], iterable: Iterable[T]) -> Iterator[List[T]]:
    """Utility batching function.

    Args:
        size: The size of the batch. If None, returns a single batch.
        iterable: The iterable to batch.

    Returns:
        An iterator over the batches.
    """
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk