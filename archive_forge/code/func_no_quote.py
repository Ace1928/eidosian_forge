import html
import html.entities
import re
from urllib.parse import quote, unquote
def no_quote(s):
    """
    Quoting that doesn't do anything
    """
    return s