import codecs
import re
from yaql.language import exceptions
@staticmethod
def t_QUOTED_VERBATIM_STRING(t):
    """
        `([^`\\\\]|\\\\.)*`
        """
    t.value = t.value[1:-1].replace('\\`', '`')
    t.type = 'QUOTED_STRING'
    return t