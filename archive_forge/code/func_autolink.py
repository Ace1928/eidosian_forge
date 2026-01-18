import copy
import re
from urllib.parse import urlsplit, unquote_plus
from lxml import etree
from lxml.html import defs
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import xhtml_to_html, _transform_result
def autolink(el, link_regexes=_link_regexes, avoid_elements=_avoid_elements, avoid_hosts=_avoid_hosts, avoid_classes=_avoid_classes):
    """
    Turn any URLs into links.

    It will search for links identified by the given regular
    expressions (by default mailto and http(s) links).

    It won't link text in an element in avoid_elements, or an element
    with a class in avoid_classes.  It won't link to anything with a
    host that matches one of the regular expressions in avoid_hosts
    (default localhost and 127.0.0.1).

    If you pass in an element, the element's tail will not be
    substituted, only the contents of the element.
    """
    if el.tag in avoid_elements:
        return
    class_name = el.get('class')
    if class_name:
        class_name = class_name.split()
        for match_class in avoid_classes:
            if match_class in class_name:
                return
    for child in list(el):
        autolink(child, link_regexes=link_regexes, avoid_elements=avoid_elements, avoid_hosts=avoid_hosts, avoid_classes=avoid_classes)
        if child.tail:
            text, tail_children = _link_text(child.tail, link_regexes, avoid_hosts, factory=el.makeelement)
            if tail_children:
                child.tail = text
                index = el.index(child)
                el[index + 1:index + 1] = tail_children
    if el.text:
        text, pre_children = _link_text(el.text, link_regexes, avoid_hosts, factory=el.makeelement)
        if pre_children:
            el.text = text
            el[:0] = pre_children