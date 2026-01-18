from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def get_typecast(self):
    """Returns the typecast or ``None`` of this object as a string."""
    midx, marker = self.token_next_by(m=(T.Punctuation, '::'))
    nidx, next_ = self.token_next(midx, skip_ws=False)
    return next_.value if next_ else None