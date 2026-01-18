from __future__ import absolute_import, division, print_function
import sys
from ansible.module_utils.common.text.converters import to_native, to_bytes
def split_routeros_command(line):
    line = to_bytes(line)
    result = []
    current = []
    index = 0
    length = len(line)
    parsing_attribute_name = False
    while index < length:
        ch = line[index:index + 1]
        index += 1
        if ch == b' ':
            if parsing_attribute_name:
                parsing_attribute_name = False
                result.append(b''.join(current))
                current = []
        elif ch == b'=' and parsing_attribute_name:
            current.append(ch)
            value, index = parse_argument_value(line, start_index=index, must_match_everything=False)
            current.append(to_bytes(value))
            parsing_attribute_name = False
            result.append(b''.join(current))
            current = []
        elif ch in (b'"', b'\\', b"'", b'=', b'(', b')', b'$', b'[', b'{', b'`', b'?'):
            raise ParseError('Found unexpected "{0}"'.format(to_native(ch)))
        else:
            current.append(ch)
            parsing_attribute_name = True
    if parsing_attribute_name and current:
        result.append(b''.join(current))
    return [to_native(part) for part in result]