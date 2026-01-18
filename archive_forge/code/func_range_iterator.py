import errno
import os
from io import BytesIO
from .lazy_import import lazy_import
import gzip
import itertools
import patiencediff
from breezy import (
from . import errors
from .i18n import gettext
def range_iterator(self):
    """Iterate through the hunks, with range indicated

        kind is "new" or "parent".
        for "new", data is a list of lines.
        for "parent", data is (parent, parent_start, parent_end)
        :return: a generator of (start, end, kind, data)
        """
    start = 0
    for hunk in self.hunks:
        if isinstance(hunk, NewText):
            kind = 'new'
            end = start + len(hunk.lines)
            data = hunk.lines
        else:
            kind = 'parent'
            start = hunk.child_pos
            end = start + hunk.num_lines
            data = (hunk.parent, hunk.parent_pos, hunk.parent_pos + hunk.num_lines)
        yield (start, end, kind, data)
        start = end