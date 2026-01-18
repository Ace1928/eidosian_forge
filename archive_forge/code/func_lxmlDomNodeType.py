from __future__ import annotations
from copy import copy, deepcopy
from xml.dom import Node
from xml.dom.minidom import Attr, NamedNodeMap
from lxml.etree import (
from lxml.html import HtmlElementClassLookup, HTMLParser
def lxmlDomNodeType(node):
    if isinstance(node, ElementBase):
        return Node.ELEMENT_NODE
    elif isinstance(node, (_ElementStringResult, _ElementUnicodeResult)):
        if node.is_attribute:
            return Node.ATTRIBUTE_NODE
        else:
            return Node.TEXT_NODE
    else:
        return Node.NOTATION_NODE