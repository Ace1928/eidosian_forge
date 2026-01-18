import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
def get_dtext(value):
    """ dtext = <printable ascii except \\ [ ]> / obs-dtext
        obs-dtext = obs-NO-WS-CTL / quoted-pair

    We allow anything except the excluded characters, but if we find any
    ASCII other than the RFC defined printable ASCII, a NonPrintableDefect is
    added to the token's defects list.  Quoted pairs are converted to their
    unquoted values, so what is returned is a ptext token, in this case a
    ValueTerminal.  If there were quoted-printables, an ObsoleteHeaderDefect is
    added to the returned token's defect list.

    """
    ptext, value, had_qp = _get_ptext_to_endchars(value, '[]')
    ptext = ValueTerminal(ptext, 'ptext')
    if had_qp:
        ptext.defects.append(errors.ObsoleteHeaderDefect('quoted printable found in domain-literal'))
    _validate_xtext(ptext)
    return (ptext, value)