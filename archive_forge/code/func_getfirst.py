from io import StringIO, BytesIO, TextIOWrapper
from collections.abc import Mapping
import sys
import os
import urllib.parse
from email.parser import FeedParser
from email.message import Message
import html
import locale
import tempfile
import warnings
def getfirst(self, key, default=None):
    """ Return the first value received."""
    if key in self:
        value = self[key]
        if isinstance(value, list):
            return value[0].value
        else:
            return value.value
    else:
        return default