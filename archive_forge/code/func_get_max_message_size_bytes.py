from __future__ import annotations
import math
from datetime import timedelta
from typing import Any, Literal, overload
from streamlit import config
from streamlit.errors import MarkdownFormattedException, StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.forward_msg_cache import populate_hash_if_needed
def get_max_message_size_bytes() -> int:
    """Returns the max websocket message size in bytes.

    This will lazyload the value from the config and store it in the global symbol table.
    """
    global _max_message_size_bytes
    if _max_message_size_bytes is None:
        _max_message_size_bytes = config.get_option('server.maxMessageSize') * int(1000000.0)
    return _max_message_size_bytes