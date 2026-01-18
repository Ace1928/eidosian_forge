from __future__ import annotations
import math
from datetime import timedelta
from typing import Any, Literal, overload
from streamlit import config
from streamlit.errors import MarkdownFormattedException, StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.forward_msg_cache import populate_hash_if_needed
def is_cacheable_msg(msg: ForwardMsg) -> bool:
    """True if the given message qualifies for caching."""
    if msg.WhichOneof('type') in {'ref_hash', 'initialize'}:
        return False
    return msg.ByteSize() >= int(config.get_option('global.minCachedMessageSize'))