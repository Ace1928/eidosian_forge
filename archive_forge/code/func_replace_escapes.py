from __future__ import annotations
import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.messages.catalog import Catalog, Message
from babel.util import _cmp, wraptext
def replace_escapes(match):
    m = match.group(1)
    if m == 'n':
        return '\n'
    elif m == 't':
        return '\t'
    elif m == 'r':
        return '\r'
    return m