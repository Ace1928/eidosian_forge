import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def html_to_xhtml(html):
    """Convert all tags in an HTML tree to XHTML by moving them to the
    XHTML namespace.
    """
    try:
        html = html.getroot()
    except AttributeError:
        pass
    prefix = '{%s}' % XHTML_NAMESPACE
    for el in html.iter(etree.Element):
        tag = el.tag
        if tag[0] != '{':
            el.tag = prefix + tag