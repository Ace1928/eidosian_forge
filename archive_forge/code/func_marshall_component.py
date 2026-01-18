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
def marshall_component(dg: DeltaGenerator, element: Element) -> Any | type[NoValue]:
    element.component_instance.component_name = self.name
    element.component_instance.form_id = current_form_id(dg)
    if self.url is not None:
        element.component_instance.url = self.url

    def marshall_element_args():
        element.component_instance.json_args = serialized_json_args
        element.component_instance.special_args.extend(special_args)
    ctx = get_script_run_ctx()
    if key is None:
        marshall_element_args()
        id = compute_widget_id('component_instance', user_key=key, name=self.name, form_id=current_form_id(dg), url=self.url, key=key, json_args=serialized_json_args, special_args=special_args, page=ctx.page_script_hash if ctx else None)
    else:
        id = compute_widget_id('component_instance', user_key=key, name=self.name, form_id=current_form_id(dg), url=self.url, key=key, page=ctx.page_script_hash if ctx else None)
    element.component_instance.id = id

    def deserialize_component(ui_value, widget_id=''):
        return ui_value
    component_state = register_widget(element_type='component_instance', element_proto=element.component_instance, user_key=key, widget_func_name=self.name, deserializer=deserialize_component, serializer=lambda x: x, ctx=ctx)
    widget_value = component_state.value
    if key is not None:
        marshall_element_args()
    if widget_value is None:
        widget_value = default
    elif isinstance(widget_value, ArrowTableProto):
        widget_value = component_arrow.arrow_proto_to_dataframe(widget_value)
    return widget_value if widget_value is not None else NoValue