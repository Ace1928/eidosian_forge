import base64
import binascii
import copy
import html.entities
import re
import xml.sax.saxutils
from .html import _cp1252
from .namespaces import _base, cc, dc, georss, itunes, mediarss, psc
from .sanitizer import _sanitize_html, _HTMLSanitizer
from .util import FeedParserDict
from .urls import _urljoin, make_safe_absolute_uri, resolve_relative_uris
@staticmethod
def looks_like_html(s):
    """
        :type s: str
        :rtype: bool
        """
    if not (re.search('</(\\w+)>', s) or re.search('&#?\\w+;', s)):
        return False
    if any((t for t in re.findall('</?(\\w+)', s) if t.lower() not in _HTMLSanitizer.acceptable_elements)):
        return False
    if any((e for e in re.findall('&(\\w+);', s) if e not in html.entities.entitydefs)):
        return False
    return True