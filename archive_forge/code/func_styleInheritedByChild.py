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
def styleInheritedByChild(node, style, nodeIsChild=False):
    """
    Returns whether 'style' is inherited by any children of the passed-in node

    If False is returned, it is guaranteed that 'style' can safely be removed
    from the passed-in node without influencing visual output of it's children

    If True is returned, the passed-in node should not have its text-based
    attributes removed.

    Warning: This method only considers presentation attributes and inline styles,
             any style sheets are ignored!
    """
    if node.nodeType != Node.ELEMENT_NODE:
        return False
    if nodeIsChild:
        if node.getAttribute(style) not in ['', 'inherit']:
            return False
        styles = _getStyle(node)
        if style in styles and (not styles[style] == 'inherit'):
            return False
    elif not node.childNodes:
        return False
    if node.childNodes:
        for child in node.childNodes:
            if styleInheritedByChild(child, style, True):
                return True
    if node.nodeName in ['a', 'defs', 'glyph', 'g', 'marker', 'mask', 'missing-glyph', 'pattern', 'svg', 'switch', 'symbol']:
        return False
    return True