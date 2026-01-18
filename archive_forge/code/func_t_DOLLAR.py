import codecs
import re
from yaql.language import exceptions
@staticmethod
def t_DOLLAR(t):
    """
        \\$\\w*
        """
    return t