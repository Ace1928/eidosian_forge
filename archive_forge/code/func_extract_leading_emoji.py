from __future__ import annotations
import re
import textwrap
from typing import TYPE_CHECKING, Any, Final, cast
from streamlit.errors import StreamlitAPIException
def extract_leading_emoji(text: str) -> tuple[str, str]:
    """Return a tuple containing the first emoji found in the given string and
    the rest of the string (minus an optional separator between the two).
    """
    if not _contains_special_chars(text):
        return ('', text)
    from streamlit.emojis import EMOJI_EXTRACTION_REGEX
    re_match = re.search(EMOJI_EXTRACTION_REGEX, text)
    if re_match is None:
        return ('', text)
    re_match: re.Match[str] = cast(Any, re_match)
    return (re_match.group(1), re_match.group(2))