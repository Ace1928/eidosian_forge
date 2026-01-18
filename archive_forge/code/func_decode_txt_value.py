from __future__ import (absolute_import, division, print_function)
import sys
import warnings
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible_collections.community.dns.plugins.module_utils.conversion.base import (
def decode_txt_value(value, character_encoding=_SENTINEL):
    """
    Given an encoded TXT value, decodes it.

    Raises DNSConversionError in case of errors.
    """
    if character_encoding is _SENTINEL:
        warnings.warn('The default value of the decode_txt_value parameter character_encoding is deprecated. Set explicitly to "octal" for the old behavior, or set to "decimal" for the new and correct behavior.', DeprecationWarning)
        character_encoding = 'octal'
    if character_encoding not in ('octal', 'decimal'):
        raise ValueError('character_encoding must be set to "octal" or "decimal"')
    value = to_bytes(value)
    state = _STATE_OUTSIDE
    index = 0
    length = len(value)
    result = []
    while index < length:
        letter = value[index:index + 1]
        index += 1
        if letter == b' ':
            if state == _STATE_QUOTED_STRING:
                result.append(letter)
            else:
                state = _STATE_OUTSIDE
        elif letter == b'\\':
            if state != _STATE_QUOTED_STRING:
                state = _STATE_UNQUOTED_STRING
            letter, index = _parse_quoted(value, index, character_encoding == 'octal')
            result.append(letter)
        elif letter == b'"':
            if state == _STATE_QUOTED_STRING:
                state = _STATE_OUTSIDE
            elif state == _STATE_OUTSIDE:
                state = _STATE_QUOTED_STRING
            else:
                raise DNSConversionError(u'Unexpected double quotation mark inside an unquoted block at position {index}'.format(index=index))
        else:
            if state != _STATE_QUOTED_STRING:
                state = _STATE_UNQUOTED_STRING
            result.append(letter)
    if state == _STATE_QUOTED_STRING:
        raise DNSConversionError(u'Missing double quotation mark at the end of value')
    return to_text(b''.join(result))