from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def group_tokens(self, grp_cls, start, end, include_end=True, extend=False):
    """Replace tokens by an instance of *grp_cls*."""
    start_idx = start
    start = self.tokens[start_idx]
    end_idx = end + include_end
    if extend and isinstance(start, grp_cls):
        subtokens = self.tokens[start_idx + 1:end_idx]
        grp = start
        grp.tokens.extend(subtokens)
        del self.tokens[start_idx + 1:end_idx]
        grp.value = text_type(start)
    else:
        subtokens = self.tokens[start_idx:end_idx]
        grp = grp_cls(subtokens)
        self.tokens[start_idx:end_idx] = [grp]
        grp.parent = self
    for token in subtokens:
        token.parent = grp
    return grp