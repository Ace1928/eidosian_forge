import codecs
import re
from yaql.language import exceptions
@staticmethod
def t_FUNC(t):
    """
        \\b[^\\W\\d]\\w*\\(
        """
    val = t.value[:-1]
    t.value = val
    return t