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
def itertext(self):
    """Create text iterator.

        The iterator loops over the element and all subelements in document
        order, returning all inner text.

        """
    tag = self.tag
    if not isinstance(tag, str) and tag is not None:
        return
    t = self.text
    if t:
        yield t
    for e in self:
        yield from e.itertext()
        t = e.tail
        if t:
            yield t