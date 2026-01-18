from __future__ import annotations
import textwrap
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Mapping
from typing_extensions import TypeAlias
from streamlit.errors import DuplicateWidgetID
from streamlit.proto.WidgetStates_pb2 import WidgetState, WidgetStates
from streamlit.runtime.state.common import (
from streamlit.type_util import ValueFieldName
def register_widget_from_metadata(metadata: WidgetMetadata[T], ctx: ScriptRunContext | None, widget_func_name: str | None, element_type: ElementType) -> RegisterWidgetResult[T]:
    """Register a widget and return its value, using an already constructed
    `WidgetMetadata`.

    This is split out from `register_widget` to allow caching code to replay
    widgets by saving and reusing the completed metadata.

    See `register_widget` for details on what this returns.
    """
    import streamlit.runtime.caching as caching
    if ctx is None:
        return RegisterWidgetResult.failure(deserializer=metadata.deserializer)
    widget_id = metadata.id
    user_key = user_key_from_widget_id(widget_id)
    if user_key is not None:
        if user_key not in ctx.widget_user_keys_this_run:
            ctx.widget_user_keys_this_run.add(user_key)
        else:
            raise DuplicateWidgetID(_build_duplicate_widget_message(widget_func_name if widget_func_name is not None else element_type, user_key))
    new_widget = widget_id not in ctx.widget_ids_this_run
    if new_widget:
        ctx.widget_ids_this_run.add(widget_id)
    else:
        raise DuplicateWidgetID(_build_duplicate_widget_message(widget_func_name if widget_func_name is not None else element_type, user_key))
    caching.save_widget_metadata(metadata)
    return ctx.session_state.register_widget(metadata, user_key)