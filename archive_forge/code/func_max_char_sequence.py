from __future__ import annotations
import re
import textwrap
from typing import TYPE_CHECKING, Any, Final, cast
from streamlit.errors import StreamlitAPIException
def max_char_sequence(string: str, char: str) -> int:
    """Returns the count of the max sequence of a given char in a string."""
    max_sequence = 0
    current_sequence = 0
    for c in string:
        if c == char:
            current_sequence += 1
            max_sequence = max(max_sequence, current_sequence)
        else:
            current_sequence = 0
    return max_sequence