import codecs
import re
from yaql.language import exceptions
@staticmethod
def t_DOUBLE_QUOTED_STRING(t):
    """
        "([^"\\\\]|\\\\.)*"
        """
    t.value = decode_escapes(t.value[1:-1])
    t.type = 'QUOTED_STRING'
    return t