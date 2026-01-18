import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
def parse_content_transfer_encoding_header(value):
    """ mechanism

    """
    cte_header = ContentTransferEncoding()
    if not value:
        cte_header.defects.append(errors.HeaderMissingRequiredValue('Missing content transfer encoding'))
        return cte_header
    try:
        token, value = get_token(value)
    except errors.HeaderParseError:
        cte_header.defects.append(errors.InvalidHeaderDefect('Expected content transfer encoding but found {!r}'.format(value)))
    else:
        cte_header.append(token)
        cte_header.cte = token.value.strip().lower()
    if not value:
        return cte_header
    while value:
        cte_header.defects.append(errors.InvalidHeaderDefect('Extra text after content transfer encoding'))
        if value[0] in PHRASE_ENDS:
            cte_header.append(ValueTerminal(value[0], 'misplaced-special'))
            value = value[1:]
        else:
            token, value = get_phrase(value)
            cte_header.append(token)
    return cte_header