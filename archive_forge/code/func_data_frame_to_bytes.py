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
def data_frame_to_bytes(df: DataFrame) -> bytes:
    """Serialize pandas.DataFrame to bytes using Apache Arrow.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe to convert.

    """
    import pyarrow as pa
    try:
        table = pa.Table.from_pandas(df)
    except (pa.ArrowTypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError) as ex:
        _LOGGER.info('Serialization of dataframe to Arrow table was unsuccessful due to: %s. Applying automatic fixes for column types to make the dataframe Arrow-compatible.', ex)
        df = fix_arrow_incompatible_column_types(df)
        table = pa.Table.from_pandas(df)
    return pyarrow_table_to_bytes(table)