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
def mayContainTextNodes(node):
    """
    Returns True if the passed-in node is probably a text element, or at least
    one of its descendants is probably a text element.

    If False is returned, it is guaranteed that the passed-in node has no
    business having text-based attributes.

    If True is returned, the passed-in node should not have its text-based
    attributes removed.
    """
    try:
        return node.mayContainTextNodes
    except AttributeError:
        pass
    result = True
    if node.nodeType != Node.ELEMENT_NODE:
        result = False
    elif node.namespaceURI != NS['SVG']:
        result = True
    elif node.nodeName in ['rect', 'circle', 'ellipse', 'line', 'polygon', 'polyline', 'path', 'image', 'stop']:
        result = False
    elif node.nodeName in ['g', 'clipPath', 'marker', 'mask', 'pattern', 'linearGradient', 'radialGradient', 'symbol']:
        result = False
        for child in node.childNodes:
            if mayContainTextNodes(child):
                result = True
    node.mayContainTextNodes = result
    return result