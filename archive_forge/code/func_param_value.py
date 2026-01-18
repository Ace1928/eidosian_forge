import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
@property
def param_value(self):
    for token in self:
        if token.token_type == 'value':
            return token.stripped_value
        if token.token_type == 'quoted-string':
            for token in token:
                if token.token_type == 'bare-quoted-string':
                    for token in token:
                        if token.token_type == 'value':
                            return token.stripped_value
    return ''