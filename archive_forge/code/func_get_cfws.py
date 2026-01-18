import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
def get_cfws(value):
    """CFWS = (1*([FWS] comment) [FWS]) / FWS

    """
    cfws = CFWSList()
    while value and value[0] in CFWS_LEADER:
        if value[0] in WSP:
            token, value = get_fws(value)
        else:
            token, value = get_comment(value)
        cfws.append(token)
    return (cfws, value)