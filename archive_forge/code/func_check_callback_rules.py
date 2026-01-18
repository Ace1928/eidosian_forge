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
def check_callback_rules(dg: DeltaGenerator, on_change: WidgetCallback | None) -> None:
    if runtime.exists() and is_in_form(dg) and (on_change is not None):
        raise StreamlitAPIException('With forms, callbacks can only be defined on the `st.form_submit_button`. Defining callbacks on other widgets inside a form is not allowed.')