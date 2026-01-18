import re
import textwrap
import email.message
from ._text import FoldedCase
def redent(value):
    """Correct for RFC822 indentation"""
    if not value or '\n' not in value:
        return value
    return textwrap.dedent(' ' * 8 + value)