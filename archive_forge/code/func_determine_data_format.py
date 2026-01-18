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
def determine_data_format(input_data: Any) -> DataFormat:
    """Determine the data format of the input data.

    Parameters
    ----------
    input_data : Any
        The input data to determine the data format of.

    Returns
    -------
    DataFormat
        The data format of the input data.
    """
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    if input_data is None:
        return DataFormat.EMPTY
    elif isinstance(input_data, pd.DataFrame):
        return DataFormat.PANDAS_DATAFRAME
    elif isinstance(input_data, np.ndarray):
        if len(input_data.shape) == 1:
            return DataFormat.NUMPY_LIST
        return DataFormat.NUMPY_MATRIX
    elif isinstance(input_data, pa.Table):
        return DataFormat.PYARROW_TABLE
    elif isinstance(input_data, pd.Series):
        return DataFormat.PANDAS_SERIES
    elif isinstance(input_data, pd.Index):
        return DataFormat.PANDAS_INDEX
    elif is_pandas_styler(input_data):
        return DataFormat.PANDAS_STYLER
    elif is_snowpark_data_object(input_data):
        return DataFormat.SNOWPARK_OBJECT
    elif is_pyspark_data_object(input_data):
        return DataFormat.PYSPARK_OBJECT
    elif isinstance(input_data, (list, tuple, set)):
        if is_list_of_scalars(input_data):
            if isinstance(input_data, tuple):
                return DataFormat.TUPLE_OF_VALUES
            if isinstance(input_data, set):
                return DataFormat.SET_OF_VALUES
            return DataFormat.LIST_OF_VALUES
        else:
            first_element = next(iter(input_data))
            if isinstance(first_element, dict):
                return DataFormat.LIST_OF_RECORDS
            if isinstance(first_element, (list, tuple, set)):
                return DataFormat.LIST_OF_ROWS
    elif isinstance(input_data, dict):
        if not input_data:
            return DataFormat.KEY_VALUE_DICT
        if len(input_data) > 0:
            first_value = next(iter(input_data.values()))
            if isinstance(first_value, dict):
                return DataFormat.COLUMN_INDEX_MAPPING
            if isinstance(first_value, (list, tuple)):
                return DataFormat.COLUMN_VALUE_MAPPING
            if isinstance(first_value, pd.Series):
                return DataFormat.COLUMN_SERIES_MAPPING
            if is_list_of_scalars(input_data.values()):
                return DataFormat.KEY_VALUE_DICT
    return DataFormat.UNKNOWN