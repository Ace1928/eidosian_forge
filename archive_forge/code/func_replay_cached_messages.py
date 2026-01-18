from __future__ import annotations
import contextlib
import hashlib
import threading
import types
from dataclasses import dataclass
from typing import (
from google.protobuf.message import Message
import streamlit as st
from streamlit import runtime, util
from streamlit.elements import NONWIDGET_ELEMENTS, WIDGETS
from streamlit.logger import get_logger
from streamlit.proto.Block_pb2 import Block
from streamlit.runtime.caching.cache_errors import (
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.hashing import update_hash
from streamlit.runtime.scriptrunner.script_run_context import (
from streamlit.runtime.state.common import WidgetMetadata
from streamlit.util import HASHLIB_KWARGS
def replay_cached_messages(result: CachedResult, cache_type: CacheType, cached_func: types.FunctionType) -> None:
    """Replay the st element function calls that happened when executing a
    cache-decorated function.

    When a cache function is executed, we record the element and block messages
    produced, and use those to reproduce the DeltaGenerator calls, so the elements
    will appear in the web app even when execution of the function is skipped
    because the result was cached.

    To make this work, for each st function call we record an identifier for the
    DG it was effectively called on (see Note [DeltaGenerator method invocation]).
    We also record the identifier for each DG returned by an st function call, if
    it returns one. Then, for each recorded message, we get the current DG instance
    corresponding to the DG the message was originally called on, and enqueue the
    message using that, recording any new DGs produced in case a later st function
    call is on one of them.
    """
    from streamlit.delta_generator import DeltaGenerator
    from streamlit.runtime.state.widgets import register_widget_from_metadata
    returned_dgs: dict[str, DeltaGenerator] = {}
    returned_dgs[result.main_id] = st._main
    returned_dgs[result.sidebar_id] = st.sidebar
    ctx = get_script_run_ctx()
    try:
        for msg in result.messages:
            if isinstance(msg, ElementMsgData):
                if msg.widget_metadata is not None:
                    register_widget_from_metadata(msg.widget_metadata.metadata, ctx, None, msg.delta_type)
                if msg.media_data is not None:
                    for data in msg.media_data:
                        runtime.get_instance().media_file_mgr.add(data.media, data.mimetype, data.media_id)
                dg = returned_dgs[msg.id_of_dg_called_on]
                maybe_dg = dg._enqueue(msg.delta_type, msg.message)
                if isinstance(maybe_dg, DeltaGenerator):
                    returned_dgs[msg.returned_dgs_id] = maybe_dg
            elif isinstance(msg, BlockMsgData):
                dg = returned_dgs[msg.id_of_dg_called_on]
                new_dg = dg._block(msg.message)
                returned_dgs[msg.returned_dgs_id] = new_dg
    except KeyError:
        raise CacheReplayClosureError(cache_type, cached_func)