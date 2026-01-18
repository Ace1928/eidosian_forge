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
def removeUnusedAttributesOnParent(elem):
    """
    This recursively calls this function on all children of the element passed in,
    then removes any unused attributes on this elem if none of the children inherit it
    """
    num = 0
    childElements = []
    for child in elem.childNodes:
        if child.nodeType == Node.ELEMENT_NODE:
            childElements.append(child)
            num += removeUnusedAttributesOnParent(child)
    if len(childElements) <= 1:
        return num
    attrList = elem.attributes
    unusedAttrs = {}
    for index in range(attrList.length):
        attr = attrList.item(index)
        if attr.nodeName in ['clip-rule', 'display-align', 'fill', 'fill-opacity', 'fill-rule', 'font', 'font-family', 'font-size', 'font-size-adjust', 'font-stretch', 'font-style', 'font-variant', 'font-weight', 'letter-spacing', 'pointer-events', 'shape-rendering', 'stroke', 'stroke-dasharray', 'stroke-dashoffset', 'stroke-linecap', 'stroke-linejoin', 'stroke-miterlimit', 'stroke-opacity', 'stroke-width', 'text-anchor', 'text-decoration', 'text-rendering', 'visibility', 'word-spacing', 'writing-mode']:
            unusedAttrs[attr.nodeName] = attr.nodeValue
    for childNum in range(len(childElements)):
        child = childElements[childNum]
        inheritedAttrs = []
        for name in unusedAttrs:
            val = child.getAttribute(name)
            if val == '' or val is None or val == 'inherit':
                inheritedAttrs.append(name)
        for a in inheritedAttrs:
            del unusedAttrs[a]
    for name in unusedAttrs:
        elem.removeAttribute(name)
        num += 1
    return num