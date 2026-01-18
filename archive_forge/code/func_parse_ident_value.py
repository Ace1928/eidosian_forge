from __future__ import (absolute_import, division, print_function)
def parse_ident_value(cur):
    if _is_ident(cur):
        value.append(cur)
        parser.next()
        return _Mode.IDENT_VALUE
    else:
        handle_kv()
        parser.next()
        return _Mode.GARBAGE