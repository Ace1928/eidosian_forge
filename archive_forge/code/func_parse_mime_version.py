import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
def parse_mime_version(value):
    """ mime-version = [CFWS] 1*digit [CFWS] "." [CFWS] 1*digit [CFWS]

    """
    mime_version = MIMEVersion()
    if not value:
        mime_version.defects.append(errors.HeaderMissingRequiredValue('Missing MIME version number (eg: 1.0)'))
        return mime_version
    if value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        mime_version.append(token)
        if not value:
            mime_version.defects.append(errors.HeaderMissingRequiredValue('Expected MIME version number but found only CFWS'))
    digits = ''
    while value and value[0] != '.' and (value[0] not in CFWS_LEADER):
        digits += value[0]
        value = value[1:]
    if not digits.isdigit():
        mime_version.defects.append(errors.InvalidHeaderDefect('Expected MIME major version number but found {!r}'.format(digits)))
        mime_version.append(ValueTerminal(digits, 'xtext'))
    else:
        mime_version.major = int(digits)
        mime_version.append(ValueTerminal(digits, 'digits'))
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        mime_version.append(token)
    if not value or value[0] != '.':
        if mime_version.major is not None:
            mime_version.defects.append(errors.InvalidHeaderDefect('Incomplete MIME version; found only major number'))
        if value:
            mime_version.append(ValueTerminal(value, 'xtext'))
        return mime_version
    mime_version.append(ValueTerminal('.', 'version-separator'))
    value = value[1:]
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        mime_version.append(token)
    if not value:
        if mime_version.major is not None:
            mime_version.defects.append(errors.InvalidHeaderDefect('Incomplete MIME version; found only major number'))
        return mime_version
    digits = ''
    while value and value[0] not in CFWS_LEADER:
        digits += value[0]
        value = value[1:]
    if not digits.isdigit():
        mime_version.defects.append(errors.InvalidHeaderDefect('Expected MIME minor version number but found {!r}'.format(digits)))
        mime_version.append(ValueTerminal(digits, 'xtext'))
    else:
        mime_version.minor = int(digits)
        mime_version.append(ValueTerminal(digits, 'digits'))
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        mime_version.append(token)
    if value:
        mime_version.defects.append(errors.InvalidHeaderDefect('Excess non-CFWS text after MIME version'))
        mime_version.append(ValueTerminal(value, 'xtext'))
    return mime_version