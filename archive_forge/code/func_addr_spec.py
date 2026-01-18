import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
@property
def addr_spec(self):
    nameset = set(self.local_part)
    if len(nameset) > len(nameset - DOT_ATOM_ENDS):
        lp = quote_string(self.local_part)
    else:
        lp = self.local_part
    if self.domain is not None:
        return lp + '@' + self.domain
    return lp