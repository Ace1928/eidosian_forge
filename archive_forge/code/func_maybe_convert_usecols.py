from __future__ import annotations
from collections.abc import (
from typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import (
def maybe_convert_usecols(usecols: str | list[int] | list[str] | usecols_func | None) -> None | list[int] | list[str] | usecols_func:
    """
    Convert `usecols` into a compatible format for parsing in `parsers.py`.

    Parameters
    ----------
    usecols : object
        The use-columns object to potentially convert.

    Returns
    -------
    converted : object
        The compatible format of `usecols`.
    """
    if usecols is None:
        return usecols
    if is_integer(usecols):
        raise ValueError('Passing an integer for `usecols` is no longer supported.  Please pass in a list of int from 0 to `usecols` inclusive instead.')
    if isinstance(usecols, str):
        return _range2cols(usecols)
    return usecols