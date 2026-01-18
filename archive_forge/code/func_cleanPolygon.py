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
def cleanPolygon(elem, options):
    """
       Remove unnecessary closing point of polygon points attribute
    """
    global _num_points_removed_from_polygon
    pts = parseListOfPoints(elem.getAttribute('points'))
    N = len(pts) / 2
    if N >= 2:
        startx, starty = pts[:2]
        endx, endy = pts[-2:]
        if startx == endx and starty == endy:
            del pts[-2:]
            _num_points_removed_from_polygon += 1
    elem.setAttribute('points', scourCoordinates(pts, options, True))