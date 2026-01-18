from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def get_token_at_offset(self, offset):
    """Returns the token that is on position offset."""
    idx = 0
    for token in self.flatten():
        end = idx + len(token.value)
        if idx <= offset < end:
            return token
        idx = end