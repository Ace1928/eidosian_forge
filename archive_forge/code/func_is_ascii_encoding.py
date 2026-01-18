import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def is_ascii_encoding(encoding: str) -> bool:
    """Checks if a given encoding is ascii."""
    try:
        return codecs.lookup(encoding).name == 'ascii'
    except LookupError:
        return False