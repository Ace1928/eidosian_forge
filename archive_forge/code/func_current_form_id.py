from __future__ import annotations
import textwrap
from typing import TYPE_CHECKING, Literal, NamedTuple, cast
from streamlit import runtime
from streamlit.errors import StreamlitAPIException
from streamlit.proto import Block_pb2
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import WidgetArgs, WidgetCallback, WidgetKwargs
def current_form_id(dg: DeltaGenerator) -> str:
    """Return the form_id for the current form, or the empty string if we're
    not inside an `st.form` block.

    (We return the empty string, instead of None, because this value is
    assigned to protobuf message fields, and None is not valid.)
    """
    form_data = _current_form(dg)
    if form_data is None:
        return ''
    return form_data.form_id