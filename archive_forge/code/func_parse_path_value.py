from __future__ import absolute_import
import re
import operator
import sys
def parse_path_value(next):
    token = next()
    value = token[0]
    if value:
        if value[:1] == "'" or value[:1] == '"':
            return value[1:-1]
        try:
            return int(value)
        except ValueError:
            pass
    elif token[1].isdigit():
        return int(token[1])
    else:
        name = token[1].lower()
        if name == 'true':
            return True
        elif name == 'false':
            return False
    raise ValueError("Invalid attribute predicate: '%s'" % value)