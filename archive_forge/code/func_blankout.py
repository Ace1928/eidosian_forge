import warnings
from io import StringIO
from django.template.base import Lexer, TokenType
from django.utils.regex_helper import _lazy_re_compile
from . import TranslatorCommentWarning, trim_whitespace
def blankout(src, char):
    """
    Change every non-whitespace character to the given char.
    Used in the templatize function.
    """
    return dot_re.sub(char, src)