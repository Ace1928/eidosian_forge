import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
def are_all_equal(iterable):
    """
    Returns ``True`` if and only if all elements in `iterable` are equal; and
    ``False`` otherwise.

    Parameters
    ----------
    iterable: collections.abc.Iterable
        The container whose elements will be checked.

    Returns
    -------
    bool
        ``True`` iff all elements in `iterable` compare equal, ``False``
        otherwise.
    """
    try:
        shape = iterable.shape
    except AttributeError:
        pass
    else:
        if len(shape) > 1:
            message = 'The function does not works on multidimensional arrays.'
            raise NotImplementedError(message) from None
    iterator = iter(iterable)
    first = next(iterator, None)
    return all((item == first for item in iterator))