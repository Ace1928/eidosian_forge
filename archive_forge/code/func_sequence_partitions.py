from collections import Counter, defaultdict, OrderedDict
from itertools import (
from itertools import product as cartes # noqa: F401
from operator import gt
from sympy.utilities.enumerative import (
from sympy.utilities.misc import as_int
from sympy.utilities.decorator import deprecated
def sequence_partitions(l, n, /):
    """Returns the partition of sequence $l$ into $n$ bins

    Explanation
    ===========

    Given the sequence $l_1 \\cdots l_m \\in V^+$ where
    $V^+$ is the Kleene plus of $V$

    The set of $n$ partitions of $l$ is defined as:

    .. math::
        \\{(s_1, \\cdots, s_n) | s_1 \\in V^+, \\cdots, s_n \\in V^+,
        s_1 \\cdots s_n = l_1 \\cdots l_m\\}

    Parameters
    ==========

    l : Sequence[T]
        A nonempty sequence of any Python objects

    n : int
        A positive integer

    Yields
    ======

    out : list[Sequence[T]]
        A list of sequences with concatenation equals $l$.
        This should conform with the type of $l$.

    Examples
    ========

    >>> from sympy.utilities.iterables import sequence_partitions
    >>> for out in sequence_partitions([1, 2, 3, 4], 2):
    ...     print(out)
    [[1], [2, 3, 4]]
    [[1, 2], [3, 4]]
    [[1, 2, 3], [4]]

    Notes
    =====

    This is modified version of EnricoGiampieri's partition generator
    from https://stackoverflow.com/questions/13131491/partition-n-items-into-k-bins-in-python-lazily

    See Also
    ========

    sequence_partitions_empty
    """
    if n == 1 and l:
        yield [l]
        return
    for i in range(1, len(l)):
        for part in sequence_partitions(l[i:], n - 1):
            yield ([l[:i]] + part)