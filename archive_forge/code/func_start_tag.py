import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def start_tag(el):
    """
    The text representation of the start tag for a tag.
    """
    return '<%s%s>' % (el.tag, ''.join([' %s="%s"' % (name, html_escape(value, True)) for name, value in el.attrib.items()]))