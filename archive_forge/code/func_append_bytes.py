import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def append_bytes(self, relpath, bytes, mode=None):
    """See Transport.append_bytes"""
    if self._range_hint is not None:
        before = self._append_by_head_put(relpath, bytes)
    else:
        before = self._append_by_get_put(relpath, bytes)
    return before