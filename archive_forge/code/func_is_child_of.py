from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def is_child_of(self, other):
    """Returns ``True`` if this token is a direct child of *other*."""
    return self.parent == other