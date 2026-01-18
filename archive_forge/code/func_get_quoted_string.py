import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
def get_quoted_string(value):
    """quoted-string = [CFWS] <bare-quoted-string> [CFWS]

    'bare-quoted-string' is an intermediate class defined by this
    parser and not by the RFC grammar.  It is the quoted string
    without any attached CFWS.
    """
    quoted_string = QuotedString()
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        quoted_string.append(token)
    token, value = get_bare_quoted_string(value)
    quoted_string.append(token)
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        quoted_string.append(token)
    return (quoted_string, value)