import collections
from numba.core import types
@wrap
def run_timsort_with_values(keys, values):
    """
        Run timsort over the given keys and values.
        """
    run_timsort_with_mergestate(merge_init_with_values(keys, values), keys, values)