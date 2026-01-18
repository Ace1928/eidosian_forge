import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def fragments_fromstring(html, no_leading_text=False, base_url=None, parser=None, **kw):
    """Parses several HTML elements, returning a list of elements.

    The first item in the list may be a string.
    If no_leading_text is true, then it will be an error if there is
    leading text, and it will always be a list of only elements.

    base_url will set the document's base_url attribute
    (and the tree's docinfo.URL).
    """
    if parser is None:
        parser = html_parser
    if isinstance(html, bytes):
        if not _looks_like_full_html_bytes(html):
            html = b'<html><body>' + html + b'</body></html>'
    elif not _looks_like_full_html_unicode(html):
        html = '<html><body>%s</body></html>' % html
    doc = document_fromstring(html, parser=parser, base_url=base_url, **kw)
    assert _nons(doc.tag) == 'html'
    bodies = [e for e in doc if _nons(e.tag) == 'body']
    assert len(bodies) == 1, 'too many bodies: %r in %r' % (bodies, html)
    body = bodies[0]
    elements = []
    if no_leading_text and body.text and body.text.strip():
        raise etree.ParserError('There is leading text: %r' % body.text)
    if body.text and body.text.strip():
        elements.append(body.text)
    elements.extend(body)
    return elements