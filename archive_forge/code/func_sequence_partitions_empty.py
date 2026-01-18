from collections import Counter, defaultdict, OrderedDict
from itertools import (
from itertools import product as cartes # noqa: F401
from operator import gt
from sympy.utilities.enumerative import (
from sympy.utilities.misc import as_int
from sympy.utilities.decorator import deprecated
def sequence_partitions_empty(l, n, /):
    """Returns the partition of sequence $l$ into $n$ bins with
    empty sequence

    Explanation
    ===========

    Given the sequence $l_1 \\cdots l_m \\in V^*$ where
    $V^*$ is the Kleene star of $V$

    The set of $n$ partitions of $l$ is defined as:

    .. math::
        \\{(s_1, \\cdots, s_n) | s_1 \\in V^*, \\cdots, s_n \\in V^*,
        s_1 \\cdots s_n = l_1 \\cdots l_m\\}

    There are more combinations than :func:`sequence_partitions` because
    empty sequence can fill everywhere, so we try to provide different
    utility for this.

    Parameters
    ==========

    l : Sequence[T]
        A sequence of any Python objects (can be possibly empty)

    n : int
        A positive integer

    Yields
    ======

    out : list[Sequence[T]]
        A list of sequences with concatenation equals $l$.
        This should conform with the type of $l$.

    Examples
    ========

    >>> from sympy.utilities.iterables import sequence_partitions_empty
    >>> for out in sequence_partitions_empty([1, 2, 3, 4], 2):
    ...     print(out)
    [[], [1, 2, 3, 4]]
    [[1], [2, 3, 4]]
    [[1, 2], [3, 4]]
    [[1, 2, 3], [4]]
    [[1, 2, 3, 4], []]

    See Also
    ========

    sequence_partitions
    """
    if n < 1:
        return
    if n == 1:
        yield [l]
        return
    for i in range(0, len(l) + 1):
        for part in sequence_partitions_empty(l[i:], n - 1):
            yield ([l[:i]] + part)