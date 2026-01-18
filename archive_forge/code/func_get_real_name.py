from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def get_real_name(self):
    """Returns the real name (object name) of this identifier."""
    dot_idx, _ = self.token_next_by(m=(T.Punctuation, '.'))
    return self._get_first_name(dot_idx, real_name=True)