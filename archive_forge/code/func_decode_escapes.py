import codecs
import re
from yaql.language import exceptions
def decode_escapes(s):

    def decode_match(match):
        return codecs.decode(match.group(0), 'unicode-escape')
    return ESCAPE_SEQUENCE_RE.sub(decode_match, s)