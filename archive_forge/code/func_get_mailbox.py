import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
def get_mailbox(value):
    """ mailbox = name-addr / addr-spec

    """
    mailbox = Mailbox()
    try:
        token, value = get_name_addr(value)
    except errors.HeaderParseError:
        try:
            token, value = get_addr_spec(value)
        except errors.HeaderParseError:
            raise errors.HeaderParseError("expected mailbox but found '{}'".format(value))
    if any((isinstance(x, errors.InvalidHeaderDefect) for x in token.all_defects)):
        mailbox.token_type = 'invalid-mailbox'
    mailbox.append(token)
    return (mailbox, value)