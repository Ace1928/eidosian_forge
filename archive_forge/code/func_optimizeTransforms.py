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
def optimizeTransforms(element, options):
    """
    Attempts to optimise transform specifications on the given node and its children.

    Returns the number of bytes saved after performing these reductions.
    """
    num = 0
    for transformAttr in ['transform', 'patternTransform', 'gradientTransform']:
        val = element.getAttribute(transformAttr)
        if val != '':
            transform = svg_transform_parser.parse(val)
            optimizeTransform(transform)
            newVal = serializeTransform(transform)
            if len(newVal) < len(val):
                if len(newVal):
                    element.setAttribute(transformAttr, newVal)
                else:
                    element.removeAttribute(transformAttr)
                num += len(val) - len(newVal)
    for child in element.childNodes:
        if child.nodeType == Node.ELEMENT_NODE:
            num += optimizeTransforms(child, options)
    return num