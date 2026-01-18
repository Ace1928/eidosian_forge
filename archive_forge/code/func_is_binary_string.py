from __future__ import annotations
import re
import textwrap
from typing import TYPE_CHECKING, Any, Final, cast
from streamlit.errors import StreamlitAPIException
def is_binary_string(inp: bytes) -> bool:
    """Guess if an input bytesarray can be encoded as a string."""
    return bool(inp.translate(None, TEXTCHARS))