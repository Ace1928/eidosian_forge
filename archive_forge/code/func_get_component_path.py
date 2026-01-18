from __future__ import annotations
import inspect
import json
import os
import threading
from typing import TYPE_CHECKING, Any, Final
import streamlit
from streamlit import type_util, util
from streamlit.elements.form import current_form_id
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.proto.Components_pb2 import ArrowTable as ArrowTableProto
from streamlit.proto.Components_pb2 import SpecialArg
from streamlit.proto.Element_pb2 import Element
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.state import NoValue, register_widget
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import to_bytes
def get_component_path(self, name: str) -> str | None:
    """Return the filesystem path for the component with the given name.

        If no such component is registered, or if the component exists but is
        being served from a URL, return None instead.
        """
    component = self._components.get(name, None)
    return component.abspath if component is not None else None