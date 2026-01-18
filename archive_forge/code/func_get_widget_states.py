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
def get_widget_states(self) -> list[WidgetStateProto]:
    """Return a list of serialized widget values for each widget with a value."""
    return self._new_widget_state.as_widget_states()