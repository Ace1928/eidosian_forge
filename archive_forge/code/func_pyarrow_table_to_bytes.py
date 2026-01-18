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
def pyarrow_table_to_bytes(table: pa.Table) -> bytes:
    """Serialize pyarrow.Table to bytes using Apache Arrow.

    Parameters
    ----------
    table : pyarrow.Table
        A table to convert.

    """
    try:
        table = _maybe_truncate_table(table)
    except RecursionError as err:
        _LOGGER.warning('Recursion error while truncating Arrow table. This is not supposed to happen.', exc_info=err)
    import pyarrow as pa
    sink = pa.BufferOutputStream()
    writer = pa.RecordBatchStreamWriter(sink, table.schema)
    writer.write_table(table)
    writer.close()
    return cast(bytes, sink.getvalue().to_pybytes())