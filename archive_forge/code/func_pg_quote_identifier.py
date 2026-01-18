from __future__ import (absolute_import, division, print_function)
import re
def pg_quote_identifier(identifier, id_type):
    identifier_fragments = _identifier_parse(identifier, quote_char='"')
    if len(identifier_fragments) > _PG_IDENTIFIER_TO_DOT_LEVEL[id_type]:
        raise SQLParseError('PostgreSQL does not support %s with more than %i dots' % (id_type, _PG_IDENTIFIER_TO_DOT_LEVEL[id_type]))
    return '.'.join(identifier_fragments)