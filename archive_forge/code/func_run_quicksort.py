import collections
import numpy as np
from numba.core import types
@wrap
def run_quicksort(A):
    return run_quicksort1(A)