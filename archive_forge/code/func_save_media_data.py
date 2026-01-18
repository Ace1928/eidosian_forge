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
def save_media_data(image_data: bytes | str, mimetype: str, image_id: str) -> None:
    CACHE_DATA_MESSAGE_REPLAY_CTX.save_image_data(image_data, mimetype, image_id)
    CACHE_RESOURCE_MESSAGE_REPLAY_CTX.save_image_data(image_data, mimetype, image_id)