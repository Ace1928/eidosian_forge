import re
import sys
from html import escape
from urllib.parse import quote
from paste.util.looper import looper
def html_quote(value):
    if value is None:
        return ''
    if not isinstance(value, str):
        value = str(value)
    value = escape(value, 1)
    return value