from __future__ import annotations
import contextlib
import copy
import math
import re
import types
from enum import Enum, EnumMeta, auto
from typing import (
from typing_extensions import TypeAlias, TypeGuard
import streamlit as st
from streamlit import config, errors
from streamlit import logger as _logger
from streamlit import string_util
from streamlit.errors import StreamlitAPIException
def convert_anything_to_df(data: Any, max_unevaluated_rows: int=MAX_UNEVALUATED_DF_ROWS, ensure_copy: bool=False, allow_styler: bool=False) -> DataFrame | Styler:
    """Try to convert different formats to a Pandas Dataframe.

    Parameters
    ----------
    data : ndarray, Iterable, dict, DataFrame, Styler, pa.Table, None, dict, list, or any

    max_unevaluated_rows: int
        If unevaluated data is detected this func will evaluate it,
        taking max_unevaluated_rows, defaults to 10k and 100 for st.table

    ensure_copy: bool
        If True, make sure to always return a copy of the data. If False, it depends on the
        type of the data. For example, a Pandas DataFrame will be returned as-is.

    allow_styler: bool
        If True, allows this to return a Pandas Styler object as well. If False, returns
        a plain Pandas DataFrame (which, of course, won't contain the Styler's styles).

    Returns
    -------
    pandas.DataFrame or pandas.Styler

    """
    import pandas as pd
    if is_type(data, _PANDAS_DF_TYPE_STR):
        return data.copy() if ensure_copy else cast(pd.DataFrame, data)
    if is_pandas_styler(data):
        sr = cast('StyleRenderer', data)
        if allow_styler:
            if ensure_copy:
                out = copy.deepcopy(sr)
                out.data = sr.data.copy()
                return cast('Styler', out)
            else:
                return data
        else:
            return cast('Styler', sr.data.copy() if ensure_copy else sr.data)
    if is_type(data, 'numpy.ndarray'):
        if len(data.shape) == 0:
            return pd.DataFrame([])
        return pd.DataFrame(data)
    if is_type(data, _SNOWPARK_DF_TYPE_STR) or is_type(data, _SNOWPARK_TABLE_TYPE_STR) or is_type(data, _PYSPARK_DF_TYPE_STR):
        if is_type(data, _PYSPARK_DF_TYPE_STR):
            data = data.limit(max_unevaluated_rows).toPandas()
        else:
            data = pd.DataFrame(data.take(max_unevaluated_rows))
        if data.shape[0] == max_unevaluated_rows:
            st.caption(f'⚠️ Showing only {string_util.simplify_number(max_unevaluated_rows)} rows. Call `collect()` on the dataframe to show more.')
        return cast(pd.DataFrame, data)
    if hasattr(data, 'to_pandas'):
        return cast(pd.DataFrame, data.to_pandas())
    try:
        return pd.DataFrame(data)
    except ValueError as ex:
        if isinstance(data, dict):
            with contextlib.suppress(ValueError):
                return pd.DataFrame.from_dict(data, orient='index')
        raise errors.StreamlitAPIException(f'\nUnable to convert object of type `{type(data)}` to `pandas.DataFrame`.\nOffending object:\n```py\n{data}\n```') from ex