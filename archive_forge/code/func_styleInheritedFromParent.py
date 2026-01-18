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
def styleInheritedFromParent(node, style):
    """
    Returns the value of 'style' that is inherited from the parents of the passed-in node

    Warning: This method only considers presentation attributes and inline styles,
             any style sheets are ignored!
    """
    parentNode = node.parentNode
    if parentNode.nodeType == Node.DOCUMENT_NODE:
        return None
    styles = _getStyle(parentNode)
    if style in styles:
        value = styles[style]
        if not value == 'inherit':
            return value
    value = parentNode.getAttribute(style)
    if value not in ['', 'inherit']:
        return parentNode.getAttribute(style)
    return styleInheritedFromParent(parentNode, style)