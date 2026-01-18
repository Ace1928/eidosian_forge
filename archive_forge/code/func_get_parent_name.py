from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def get_parent_name(self):
    """Return name of the parent object if any.

        A parent object is identified by the first occurring dot.
        """
    dot_idx, _ = self.token_next_by(m=(T.Punctuation, '.'))
    _, prev_ = self.token_prev(dot_idx)
    return remove_quotes(prev_.value) if prev_ is not None else None