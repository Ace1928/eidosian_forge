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
def declare_component(name: str, path: str | None=None, url: str | None=None) -> CustomComponent:
    """Create and register a custom component.

    Parameters
    ----------
    name: str
        A short, descriptive name for the component. Like, "slider".
    path: str or None
        The path to serve the component's frontend files from. Either
        `path` or `url` must be specified, but not both.
    url: str or None
        The URL that the component is served from. Either `path` or `url`
        must be specified, but not both.

    Returns
    -------
    CustomComponent
        A CustomComponent that can be called like a function.
        Calling the component will create a new instance of the component
        in the Streamlit app.

    """
    current_frame = inspect.currentframe()
    assert current_frame is not None
    caller_frame = current_frame.f_back
    assert caller_frame is not None
    module = inspect.getmodule(caller_frame)
    assert module is not None
    module_name = module.__name__
    if module_name == '__main__':
        file_path = inspect.getfile(caller_frame)
        filename = os.path.basename(file_path)
        module_name, _ = os.path.splitext(filename)
    component_name = f'{module_name}.{name}'
    component = CustomComponent(name=component_name, path=path, url=url)
    ComponentRegistry.instance().register_component(component)
    return component