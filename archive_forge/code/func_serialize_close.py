from __future__ import annotations
import struct
from typing import Any, Awaitable, Callable, NamedTuple, Optional, Sequence, Tuple
from .. import extensions, frames
from ..exceptions import PayloadTooBig, ProtocolError
from ..frames import (  # noqa: E402, F401, I001
def serialize_close(code: int, reason: str) -> bytes:
    """
    Serialize the payload for a close frame.

    """
    return Close(code, reason).serialize()