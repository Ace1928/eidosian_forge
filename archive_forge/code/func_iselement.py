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
def iselement(element):
    """Return True if *element* appears to be an Element."""
    return hasattr(element, 'tag')