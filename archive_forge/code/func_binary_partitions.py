from collections import Counter, defaultdict, OrderedDict
from itertools import (
from itertools import product as cartes # noqa: F401
from operator import gt
from sympy.utilities.enumerative import (
from sympy.utilities.misc import as_int
from sympy.utilities.decorator import deprecated
def binary_partitions(n):
    """
    Generates the binary partition of n.

    A binary partition consists only of numbers that are
    powers of two. Each step reduces a `2^{k+1}` to `2^k` and
    `2^k`. Thus 16 is converted to 8 and 8.

    Examples
    ========

    >>> from sympy.utilities.iterables import binary_partitions
    >>> for i in binary_partitions(5):
    ...     print(i)
    ...
    [4, 1]
    [2, 2, 1]
    [2, 1, 1, 1]
    [1, 1, 1, 1, 1]

    References
    ==========

    .. [1] TAOCP 4, section 7.2.1.5, problem 64

    """
    from math import ceil, log
    power = int(2 ** ceil(log(n, 2)))
    acc = 0
    partition = []
    while power:
        if acc + power <= n:
            partition.append(power)
            acc += power
        power >>= 1
    last_num = len(partition) - 1 - (n & 1)
    while last_num >= 0:
        yield partition
        if partition[last_num] == 2:
            partition[last_num] = 1
            partition.append(1)
            last_num -= 1
            continue
        partition.append(1)
        partition[last_num] >>= 1
        x = partition[last_num + 1] = partition[last_num]
        last_num += 1
        while x > 1:
            if x <= len(partition) - last_num - 1:
                del partition[-x + 1:]
                last_num += 1
                partition[last_num] = x
            else:
                x >>= 1
    yield ([1] * n)