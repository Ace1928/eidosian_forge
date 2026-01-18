from __future__ import annotations
import re
from traitlets import List, Unicode
from .base import Preprocessor
def check_conditions(self, cell):
    """
        Checks that a cell matches the pattern.

        Returns: Boolean.
        True means cell should *not* be removed.
        """
    pattern = re.compile('|'.join(('(?:%s)' % pattern for pattern in self.patterns)))
    return not pattern.match(cell.source)