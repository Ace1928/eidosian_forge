from __future__ import annotations
import re
import textwrap
from typing import TYPE_CHECKING, Any, Final, cast
from streamlit.errors import StreamlitAPIException
def simplify_number(num: int) -> str:
    """Simplifies number into Human readable format, returns str"""
    num_converted = float(f'{num:.2g}')
    magnitude = 0
    while abs(num_converted) >= 1000:
        magnitude += 1
        num_converted /= 1000.0
    return '{}{}'.format(f'{num_converted:f}'.rstrip('0').rstrip('.'), ['', 'k', 'm', 'b', 't'][magnitude])