import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def serialize_html_fragment(el, skip_outer=False):
    """ Serialize a single lxml element as HTML.  The serialized form
    includes the elements tail.  

    If skip_outer is true, then don't serialize the outermost tag
    """
    assert not isinstance(el, basestring), 'You should pass in an element, not a string like %r' % el
    html = etree.tostring(el, method='html', encoding=_unicode)
    if skip_outer:
        html = html[html.find('>') + 1:]
        html = html[:html.rfind('<')]
        return html.strip()
    else:
        return html