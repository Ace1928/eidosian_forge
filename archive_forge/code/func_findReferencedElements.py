from __future__ import division         # use "true" division instead of integer division in Python 2 (see PEP 238)
from __future__ import print_function   # use print() as a function in Python 2 (see PEP 3105)
from __future__ import absolute_import  # use absolute imports by default in Python 2 (see PEP 328)
import math
import optparse
import os
import re
import sys
import time
import xml.dom.minidom
from xml.dom import Node, NotFoundErr
from collections import namedtuple, defaultdict
from decimal import Context, Decimal, InvalidOperation, getcontext
import six
from six.moves import range, urllib
from scour.svg_regex import svg_parser
from scour.svg_transform import svg_transform_parser
from scour.yocto_css import parseCssString
from scour import __version__
def findReferencedElements(node, ids=None):
    """
    Returns IDs of all referenced elements
    - node is the node at which to start the search.
    - returns a map which has the id as key and
      each value is is a set of nodes

    Currently looks at 'xlink:href' and all attributes in 'referencingProps'
    """
    global referencingProps
    if ids is None:
        ids = {}
    if node.nodeName == 'style' and node.namespaceURI == NS['SVG']:
        stylesheet = ''.join((child.nodeValue for child in node.childNodes))
        if stylesheet != '':
            cssRules = parseCssString(stylesheet)
            for rule in cssRules:
                for propname in rule['properties']:
                    propval = rule['properties'][propname]
                    findReferencingProperty(node, propname, propval, ids)
        return ids
    href = node.getAttributeNS(NS['XLINK'], 'href')
    if href != '' and len(href) > 1 and (href[0] == '#'):
        id = href[1:]
        if id in ids:
            ids[id].add(node)
        else:
            ids[id] = {node}
    styles = node.getAttribute('style').split(';')
    for style in styles:
        propval = style.split(':')
        if len(propval) == 2:
            prop = propval[0].strip()
            val = propval[1].strip()
            findReferencingProperty(node, prop, val, ids)
    for attr in referencingProps:
        val = node.getAttribute(attr).strip()
        if not val:
            continue
        findReferencingProperty(node, attr, val, ids)
    if node.hasChildNodes():
        for child in node.childNodes:
            if child.nodeType == Node.ELEMENT_NODE:
                findReferencedElements(child, ids)
    return ids