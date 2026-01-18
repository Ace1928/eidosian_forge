from __future__ import annotations
import functools
import inspect
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.core.util.numba_ import (
@functools.cache
def generate_numba_transform_func(func: Callable[..., np.ndarray], nopython: bool, nogil: bool, parallel: bool) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Any], np.ndarray]:
    """
    Generate a numba jitted transform function specified by values from engine_kwargs.

    1. jit the user's function
    2. Return a groupby transform function with the jitted function inline

    Configurations specified in engine_kwargs apply to both the user's
    function _AND_ the groupby evaluation loop.

    Parameters
    ----------
    func : function
        function to be applied to each window and will be JITed
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit

    Returns
    -------
    Numba function
    """
    numba_func = jit_user_function(func)
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency('numba')

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def group_transform(values: np.ndarray, index: np.ndarray, begin: np.ndarray, end: np.ndarray, num_columns: int, *args: Any) -> np.ndarray:
        assert len(begin) == len(end)
        num_groups = len(begin)
        result = np.empty((len(values), num_columns))
        for i in numba.prange(num_groups):
            group_index = index[begin[i]:end[i]]
            for j in numba.prange(num_columns):
                group = values[begin[i]:end[i], j]
                result[begin[i]:end[i], j] = numba_func(group, group_index, *args)
        return result
    return group_transform