from __future__ import annotations
import types
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.errors import NumbaUtilError
def get_jit_arguments(engine_kwargs: dict[str, bool] | None=None, kwargs: dict | None=None) -> dict[str, bool]:
    """
    Return arguments to pass to numba.JIT, falling back on pandas default JIT settings.

    Parameters
    ----------
    engine_kwargs : dict, default None
        user passed keyword arguments for numba.JIT
    kwargs : dict, default None
        user passed keyword arguments to pass into the JITed function

    Returns
    -------
    dict[str, bool]
        nopython, nogil, parallel

    Raises
    ------
    NumbaUtilError
    """
    if engine_kwargs is None:
        engine_kwargs = {}
    nopython = engine_kwargs.get('nopython', True)
    if kwargs and nopython:
        raise NumbaUtilError('numba does not support kwargs with nopython=True: https://github.com/numba/numba/issues/2916')
    nogil = engine_kwargs.get('nogil', False)
    parallel = engine_kwargs.get('parallel', False)
    return {'nopython': nopython, 'nogil': nogil, 'parallel': parallel}