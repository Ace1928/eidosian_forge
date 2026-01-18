from __future__ import annotations
from copy import copy, deepcopy
from xml.dom import Node
from xml.dom.minidom import Attr, NamedNodeMap
from lxml.etree import (
from lxml.html import HtmlElementClassLookup, HTMLParser
@property
def childNodes(self):
    if self.text:
        yield DomTextNode(self.text)
    for n in self.iterchildren():
        yield n
        if n.tail:
            yield DomTextNode(n.tail)