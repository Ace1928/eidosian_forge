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
def convertColor(value):
    """
       Converts the input color string and returns a #RRGGBB (or #RGB if possible) string
    """
    s = value
    if s in colors:
        s = colors[s]
    rgbpMatch = rgbp.match(s)
    if rgbpMatch is not None:
        r = int(float(rgbpMatch.group(1)) * 255.0 / 100.0)
        g = int(float(rgbpMatch.group(2)) * 255.0 / 100.0)
        b = int(float(rgbpMatch.group(3)) * 255.0 / 100.0)
        s = '#%02x%02x%02x' % (r, g, b)
    else:
        rgbMatch = rgb.match(s)
        if rgbMatch is not None:
            r = int(rgbMatch.group(1))
            g = int(rgbMatch.group(2))
            b = int(rgbMatch.group(3))
            s = '#%02x%02x%02x' % (r, g, b)
    if s[0] == '#':
        s = s.lower()
        if len(s) == 7 and s[1] == s[2] and (s[3] == s[4]) and (s[5] == s[6]):
            s = '#' + s[1] + s[3] + s[5]
    return s