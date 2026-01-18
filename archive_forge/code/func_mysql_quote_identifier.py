from __future__ import (absolute_import, division, print_function)
import re
def mysql_quote_identifier(identifier, id_type):
    identifier_fragments = _identifier_parse(identifier, quote_char='`')
    if len(identifier_fragments) - 1 > _MYSQL_IDENTIFIER_TO_DOT_LEVEL[id_type]:
        raise SQLParseError('MySQL does not support %s with more than %i dots' % (id_type, _MYSQL_IDENTIFIER_TO_DOT_LEVEL[id_type]))
    special_cased_fragments = []
    for fragment in identifier_fragments:
        if fragment == '`*`':
            special_cased_fragments.append('*')
        else:
            special_cased_fragments.append(fragment)
    return '.'.join(special_cased_fragments)