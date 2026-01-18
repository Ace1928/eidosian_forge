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
def ensure_iterable(obj: OptionSequence[V_co] | Iterable[V_co]) -> Iterable[Any]:
    """Try to convert different formats to something iterable. Most inputs
    are assumed to be iterable, but if we have a DataFrame, we can just
    select the first column to iterate over. If the input is not iterable,
    a TypeError is raised.

    Parameters
    ----------
    obj : list, tuple, numpy.ndarray, pandas.Series, pandas.DataFrame, pyspark.sql.DataFrame, snowflake.snowpark.dataframe.DataFrame or snowflake.snowpark.table.Table

    Returns
    -------
    iterable

    """
    if is_snowpark_or_pyspark_data_object(obj):
        obj = convert_anything_to_df(obj)
    if is_dataframe(obj):
        return cast(Iterable[Any], obj.iloc[:, 0])
    if is_iterable(obj):
        return obj
    raise TypeError(f'Object is not an iterable and could not be converted to one. Object: {obj}')