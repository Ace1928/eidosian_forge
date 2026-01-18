import itertools
from typing import Iterable, TypeVar
def chunk_iterable(iterable: Iterable[T], chunk_size: int) -> Iterable[Iterable[T]]:
    """Iterates over an iterator chunk by chunk.

    Taken from https://stackoverflow.com/a/8998040.
    See also https://github.com/huggingface/huggingface_hub/pull/920#discussion_r938793088.

    Args:
        iterable (`Iterable`):
            The iterable on which we want to iterate.
        chunk_size (`int`):
            Size of the chunks. Must be a strictly positive integer (e.g. >0).

    Example:

    ```python
    >>> from huggingface_hub.utils import chunk_iterable

    >>> for items in chunk_iterable(range(17), chunk_size=8):
    ...     print(items)
    # [0, 1, 2, 3, 4, 5, 6, 7]
    # [8, 9, 10, 11, 12, 13, 14, 15]
    # [16] # smaller last chunk
    ```

    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If `chunk_size` <= 0.

    <Tip warning={true}>
        The last chunk can be smaller than `chunk_size`.
    </Tip>
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError('`chunk_size` must be a strictly positive integer (>0).')
    iterator = iter(iterable)
    while True:
        try:
            next_item = next(iterator)
        except StopIteration:
            return
        yield itertools.chain((next_item,), itertools.islice(iterator, chunk_size - 1))