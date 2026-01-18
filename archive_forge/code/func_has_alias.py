from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def has_alias(self):
    """Returns ``True`` if an alias is present."""
    return self.get_alias() is not None