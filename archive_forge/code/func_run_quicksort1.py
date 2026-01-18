import collections
import numpy as np
from numba.core import types
@wrap
def run_quicksort1(A):
    R = make_res(A)
    if len(A) < 2:
        return R
    stack = [Partition(zero, zero)] * MAX_STACK
    stack[0] = Partition(zero, len(A) - 1)
    n = 1
    while n > 0:
        n -= 1
        low, high = stack[n]
        while high - low >= SMALL_QUICKSORT:
            assert n < MAX_STACK
            i = partition(A, R, low, high)
            if high - i > i - low:
                if high > i:
                    stack[n] = Partition(i + 1, high)
                    n += 1
                high = i - 1
            else:
                if i > low:
                    stack[n] = Partition(low, i - 1)
                    n += 1
                low = i + 1
        insertion_sort(A, R, low, high)
    return R