import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
@property
def all_mailboxes(self):
    if self[2].token_type != 'group-list':
        return []
    return self[2].all_mailboxes