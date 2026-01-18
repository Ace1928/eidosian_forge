import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars
def check_parse_complete(self):
    """Each text column should have been completely seen."""
    last = self.bottom - 1
    for col in range(self.right):
        if self.done[col] != last:
            return False
    return True