from typing import (
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_size
def slice_iterable(it: Union[Iterable[T], Iterator[T]], slicer: Callable[[int, T, Optional[T]], bool]) -> 'Iterable[_SliceIterable[T]]':
    """Slice the original iterable into slices by slicer

    :param it: underlying iterable
    :param slicer: taking in current number, current value, last value,
        it decides if it's a new slice

    :yield: an iterable of iterables (_SliceIterable[T])
    """
    si = _SliceIterable(it, slicer)
    while si._state < 3:
        yield si
        si.recycle()