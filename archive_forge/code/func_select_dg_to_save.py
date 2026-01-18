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
def select_dg_to_save(self, invoked_id: str, acting_on_id: str) -> str:
    """Select the id of the DG that this message should be invoked on
        during message replay.

        See Note [DeltaGenerator method invocation]

        invoked_id is the DG the st function was called on, usually `st._main`.
        acting_on_id is the DG the st function ultimately runs on, which may be different
        if the invoked DG delegated to another one because it was in a `with` block.
        """
    if len(self._seen_dg_stack) > 0 and acting_on_id in self._seen_dg_stack[-1]:
        return acting_on_id
    else:
        return invoked_id