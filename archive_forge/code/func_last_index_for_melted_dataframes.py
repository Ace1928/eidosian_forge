from __future__ import annotations
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING, Any, Hashable, Iterable, Sequence, cast, overload
import streamlit
from streamlit import config, runtime, type_util
from streamlit.elements.form import is_in_form
from streamlit.errors import StreamlitAPIException
from streamlit.proto.LabelVisibilityMessage_pb2 import LabelVisibilityMessage
from streamlit.runtime.state import WidgetCallback, get_session_state
from streamlit.runtime.state.common import RegisterWidgetResult
from streamlit.type_util import T
def last_index_for_melted_dataframes(data: DataFrameCompatible | Any) -> Hashable | None:
    if type_util.is_dataframe_compatible(data):
        data = type_util.convert_anything_to_df(data)
        if data.index.size > 0:
            return cast(Hashable, data.index[-1])
    return None