import sys
import re
import warnings
import io
import collections
import collections.abc
import contextlib
import weakref
from . import ElementPath
fromstring = XML
def getroot(self):
    """Return root element of this tree."""
    return self._root