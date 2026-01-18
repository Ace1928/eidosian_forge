from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def get_array_indices(self):
    """Returns an iterator of index token lists"""
    for token in self.tokens:
        if isinstance(token, SquareBrackets):
            yield token.tokens[1:-1]