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
def register_component(self, component: CustomComponent) -> None:
    """Register a CustomComponent.

        Parameters
        ----------
        component : CustomComponent
            The component to register.
        """
    abspath = component.abspath
    if abspath is not None and (not os.path.isdir(abspath)):
        raise StreamlitAPIException(f"No such component directory: '{abspath}'")
    with self._lock:
        existing = self._components.get(component.name)
        self._components[component.name] = component
    if existing is not None and component != existing:
        _LOGGER.warning('%s overriding previously-registered %s', component, existing)
    _LOGGER.debug('Registered component %s', component)