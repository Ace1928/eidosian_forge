import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
@property
def as_ew_allowed(self):
    """True if all top level tokens of this part may be RFC2047 encoded."""
    return all((part.as_ew_allowed for part in self))