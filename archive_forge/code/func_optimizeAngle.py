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
def optimizeAngle(angle):
    """
    Because any rotation can be expressed within 360 degrees
    of any given number, and since negative angles sometimes
    are one character longer than corresponding positive angle,
    we shorten the number to one in the range to [-90, 270[.
    """
    if angle < 0:
        angle %= -360
    else:
        angle %= 360
    if angle >= 270:
        angle -= 360
    elif angle < -90:
        angle += 360
    return angle