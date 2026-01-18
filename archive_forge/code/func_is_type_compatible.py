from __future__ import annotations
import json
from enum import Enum
from typing import TYPE_CHECKING, Dict, Final, Literal, Mapping, Union
from typing_extensions import TypeAlias
from streamlit.elements.lib.column_types import ColumnConfig, ColumnType
from streamlit.elements.lib.dicttools import remove_none_values
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
from streamlit.type_util import DataFormat, is_colum_type_arrow_incompatible
def is_type_compatible(column_type: ColumnType, data_kind: ColumnDataKind) -> bool:
    """Check if the column type is compatible with the underlying data kind.

    This check only applies to editable column types (e.g. number or text).
    Non-editable column types (e.g. bar_chart or image) can be configured for
    all data kinds (this might change in the future).

    Parameters
    ----------
    column_type : ColumnType
        The column type to check.

    data_kind : ColumnDataKind
        The data kind to check.

    Returns
    -------
    bool
        True if the column type is compatible with the data kind, False otherwise.
    """
    if column_type not in _EDITING_COMPATIBILITY_MAPPING:
        return True
    return data_kind in _EDITING_COMPATIBILITY_MAPPING[column_type]