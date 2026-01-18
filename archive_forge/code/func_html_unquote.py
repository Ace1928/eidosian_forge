import html
import html.entities
import re
from urllib.parse import quote, unquote
def html_unquote(s, encoding=None):
    """
    Decode the value.

    """
    if isinstance(s, bytes):
        s = s.decode(encoding or default_encoding)
    return _unquote_re.sub(_entity_subber, s)