import collections
from numba.core import types
@wrap
def run_timsort(keys):
    """
        Run timsort over the given keys.
        """
    values = keys
    run_timsort_with_mergestate(merge_init(keys), keys, values)