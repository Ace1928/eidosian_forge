from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def get_sublists(self):
    for token in self.tokens:
        if token.is_group:
            yield token