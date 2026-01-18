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
def on_script_finished(self, widget_ids_this_run: set[str]) -> None:
    """Called by ScriptRunner after its script finishes running.
         Updates widgets to prepare for the next script run.

        Parameters
        ----------
        widget_ids_this_run: set[str]
            The IDs of the widgets that were accessed during the script
            run. Any widget state whose ID does *not* appear in this set
            is considered "stale" and will be removed.
        """
    self._reset_triggers()
    self._remove_stale_widgets(widget_ids_this_run)