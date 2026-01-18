import re
from urllib.parse import quote
from html import _replace_charref
def safe_entity(s: str):
    """Escape characters for safety."""
    return escape(unescape(s))