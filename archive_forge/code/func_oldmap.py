from __future__ import division, absolute_import, print_function
from itertools import chain, starmap
import itertools       # since zip_longest doesn't exist on Py2
from past.types import basestring
from past.utils import PY3
def oldmap(func, *iterables):
    """
        map(function, sequence[, sequence, ...]) -> list

        Return a list of the results of applying the function to the
        items of the argument sequence(s).  If more than one sequence is
        given, the function is called with an argument list consisting of
        the corresponding item of each sequence, substituting None for
        missing values when not all sequences have the same length.  If
        the function is None, return a list of the items of the sequence
        (or a list of tuples if more than one sequence).

        Test cases:
        >>> oldmap(None, 'hello world')
        ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']

        >>> oldmap(None, range(4))
        [0, 1, 2, 3]

        More test cases are in test_past.test_builtins.
        """
    zipped = itertools.zip_longest(*iterables)
    l = list(zipped)
    if len(l) == 0:
        return []
    if func is None:
        result = l
    else:
        result = list(starmap(func, l))
    try:
        if max([len(item) for item in result]) == 1:
            return list(chain.from_iterable(result))
    except TypeError as e:
        pass
    return result