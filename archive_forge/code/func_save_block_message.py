from __future__ import annotations
import contextlib
from typing import Any, Iterator
from google.protobuf.message import Message
from streamlit.proto.Block_pb2 import Block
from streamlit.runtime.caching.cache_data_api import (
from streamlit.runtime.caching.cache_errors import CACHE_DOCS_URL
from streamlit.runtime.caching.cache_resource_api import (
from streamlit.runtime.state.common import WidgetMetadata
from streamlit.runtime.caching.cache_data_api import get_data_cache_stats_provider
from streamlit.runtime.caching.cache_resource_api import (
def save_block_message(block_proto: Block, invoked_dg_id: str, used_dg_id: str, returned_dg_id: str) -> None:
    """Save the message for a block to a thread-local callstack, so it can
    be used later to replay the block when a cache-decorated function's
    execution is skipped.
    """
    CACHE_DATA_MESSAGE_REPLAY_CTX.save_block_message(block_proto, invoked_dg_id, used_dg_id, returned_dg_id)
    CACHE_RESOURCE_MESSAGE_REPLAY_CTX.save_block_message(block_proto, invoked_dg_id, used_dg_id, returned_dg_id)