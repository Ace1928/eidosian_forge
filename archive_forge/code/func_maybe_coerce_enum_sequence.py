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
def maybe_coerce_enum_sequence(register_widget_result, options, opt_sequence):
    """Maybe Coerce a RegisterWidgetResult with a sequence of Enum members as value
    to RegisterWidgetResult[Sequence[option]] if option is an EnumType, otherwise just return
    the original RegisterWidgetResult."""
    if not all((isinstance(val, Enum) for val in register_widget_result.value)):
        return register_widget_result
    coerce_class: EnumMeta | None
    if isinstance(options, EnumMeta):
        coerce_class = options
    else:
        coerce_class = _extract_common_class_from_iter(opt_sequence)
        if coerce_class is None:
            return register_widget_result
    return RegisterWidgetResult(type(register_widget_result.value)((type_util.coerce_enum(val, coerce_class) for val in register_widget_result.value)), register_widget_result.value_changed)