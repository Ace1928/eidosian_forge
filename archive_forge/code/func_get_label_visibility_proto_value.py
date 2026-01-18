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
def get_label_visibility_proto_value(label_visibility_string: type_util.LabelVisibility) -> LabelVisibilityMessage.LabelVisibilityOptions.ValueType:
    """Returns one of LabelVisibilityMessage enum constants.py based on string value."""
    if label_visibility_string == 'visible':
        return LabelVisibilityMessage.LabelVisibilityOptions.VISIBLE
    elif label_visibility_string == 'hidden':
        return LabelVisibilityMessage.LabelVisibilityOptions.HIDDEN
    elif label_visibility_string == 'collapsed':
        return LabelVisibilityMessage.LabelVisibilityOptions.COLLAPSED
    raise ValueError(f'Unknown label visibility value: {label_visibility_string}')