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
def parseListOfPoints(s):
    """
       Parse string into a list of points.

       Returns a list containing an even number of coordinate strings
    """
    i = 0
    ws_nums = RE_COMMA_WSP.split(s.strip())
    nums = []
    for i in range(len(ws_nums)):
        negcoords = ws_nums[i].split('-')
        if len(negcoords) == 1:
            nums.append(negcoords[0])
        else:
            for j in range(len(negcoords)):
                if j == 0:
                    if negcoords[0] != '':
                        nums.append(negcoords[0])
                else:
                    prev = ''
                    if len(nums):
                        prev = nums[len(nums) - 1]
                    if prev and prev[len(prev) - 1] in ['e', 'E']:
                        nums[len(nums) - 1] = prev + '-' + negcoords[j]
                    else:
                        nums.append('-' + negcoords[j])
    if len(nums) % 2 != 0:
        return []
    i = 0
    while i < len(nums):
        try:
            nums[i] = getcontext().create_decimal(nums[i])
            nums[i + 1] = getcontext().create_decimal(nums[i + 1])
        except InvalidOperation:
            return []
        i += 2
    return nums