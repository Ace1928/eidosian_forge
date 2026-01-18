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
def controlPoints(cmd, data):
    """
       Checks if there are control points in the path data

       Returns the indices of all values in the path data which are control points
    """
    cmd = cmd.lower()
    if cmd in ['c', 's', 'q']:
        indices = range(len(data))
        if cmd == 'c':
            return [index for index in indices if index % 6 < 4]
        elif cmd in ['s', 'q']:
            return [index for index in indices if index % 4 < 2]
    return []