from __future__ import annotations
import re
import textwrap
from typing import TYPE_CHECKING, Any, Final, cast
from streamlit.errors import StreamlitAPIException
def probably_contains_html_tags(s: str) -> bool:
    """Returns True if the given string contains what seem to be HTML tags.

    Note that false positives/negatives are possible, so this function should not be
    used in contexts where complete correctness is required."""
    return bool(_RE_CONTAINS_HTML.search(s))