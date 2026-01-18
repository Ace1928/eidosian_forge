from __future__ import absolute_import, division, print_function
import sys
from ansible.module_utils.common.text.converters import to_native, to_bytes
def parse_argument_value(line, start_index=0, must_match_everything=True):
    """
    Parse an argument value (quoted or not quoted) from ``line``.

    Will start at offset ``start_index``. Returns pair ``(parsed_value,
    end_index)``, where ``end_index`` is the first character after the
    attribute.

    If ``must_match_everything`` is ``True`` (default), will fail if
    ``end_index < len(line)``.
    """
    line = to_bytes(line)
    length = len(line)
    index = start_index
    if index == length:
        raise ParseError('Expected value, but found end of string')
    quoted = False
    if line[index:index + 1] == b'"':
        quoted = True
        index += 1
    current = []
    while index < length:
        ch = line[index:index + 1]
        index += 1
        if not quoted and ch == b' ':
            index -= 1
            break
        elif ch == b'"':
            if quoted:
                quoted = False
                if line[index:index + 1] not in (b'', b' '):
                    raise ParseError('Ending \'"\' must be followed by space or end of string')
                break
            raise ParseError('\'"\' must not appear in an unquoted value')
        elif ch == b'\\':
            if not quoted:
                raise ParseError('Escape sequences can only be used inside double quotes')
            if index == length:
                raise ParseError("'\\' must not be at the end of the line")
            ch = line[index:index + 1]
            index += 1
            if ch in ESCAPE_SEQUENCES:
                current.append(ESCAPE_SEQUENCES[ch])
            else:
                d1 = ESCAPE_DIGITS.find(ch)
                if d1 < 0:
                    raise ParseError("Invalid escape sequence '\\{0}'".format(to_native(ch)))
                if index == length:
                    raise ParseError('Hex escape sequence cut off at end of line')
                ch2 = line[index:index + 1]
                d2 = ESCAPE_DIGITS.find(ch2)
                index += 1
                if d2 < 0:
                    raise ParseError("Invalid hex escape sequence '\\{0}'".format(to_native(ch + ch2)))
                current.append(_int_to_byte(d1 * 16 + d2))
        else:
            if not quoted and ch in (b"'", b'=', b'(', b')', b'$', b'[', b'{', b'`'):
                raise ParseError('"{0}" can only be used inside double quotes'.format(to_native(ch)))
            if ch == b'?':
                raise ParseError('"{0}" can only be used in escaped form'.format(to_native(ch)))
            current.append(ch)
    if quoted:
        raise ParseError('Unexpected end of string during escaped parameter')
    if must_match_everything and index < length:
        raise ParseError('Unexpected data at end of value')
    return (to_native(b''.join(current)), index)