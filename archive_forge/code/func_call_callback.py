from __future__ import annotations
import json
import pickle
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import (
from typing_extensions import TypeAlias
import streamlit as st
from streamlit import config, util
from streamlit.errors import StreamlitAPIException, UnserializableSessionStateError
from streamlit.proto.WidgetStates_pb2 import WidgetState as WidgetStateProto
from streamlit.proto.WidgetStates_pb2 import WidgetStates as WidgetStatesProto
from streamlit.runtime.state.common import (
from streamlit.runtime.state.query_params import QueryParams
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.type_util import ValueFieldName, is_array_value_field_name
def call_callback(self, widget_id: str) -> None:
    """Call the given widget's callback and return the callback's
        return value. If the widget has no callback, return None.

        If the widget doesn't exist, raise an Exception.
        """
    metadata = self.widget_metadata.get(widget_id)
    assert metadata is not None
    callback = metadata.callback
    if callback is None:
        return
    args = metadata.callback_args or ()
    kwargs = metadata.callback_kwargs or {}
    callback(*args, **kwargs)