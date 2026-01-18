import codecs
import re
from yaql.language import exceptions
@staticmethod
def t_QUOTED_STRING(t):
    """
        '([^'\\\\]|\\\\.)*'
        """
    t.value = decode_escapes(t.value[1:-1])
    return t