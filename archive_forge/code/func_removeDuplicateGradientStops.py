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
def removeDuplicateGradientStops(doc):
    global _num_elements_removed
    num = 0
    for gradType in ['linearGradient', 'radialGradient']:
        for grad in doc.getElementsByTagName(gradType):
            stops = {}
            stopsToRemove = []
            for stop in grad.getElementsByTagName('stop'):
                offsetU = SVGLength(stop.getAttribute('offset'))
                if offsetU.units == Unit.PCT:
                    offset = offsetU.value / 100.0
                elif offsetU.units == Unit.NONE:
                    offset = offsetU.value
                else:
                    offset = 0
                if int(offset) == offset:
                    stop.setAttribute('offset', str(int(offset)))
                else:
                    stop.setAttribute('offset', str(offset))
                color = stop.getAttribute('stop-color')
                opacity = stop.getAttribute('stop-opacity')
                style = stop.getAttribute('style')
                if offset in stops:
                    oldStop = stops[offset]
                    if oldStop[0] == color and oldStop[1] == opacity and (oldStop[2] == style):
                        stopsToRemove.append(stop)
                stops[offset] = [color, opacity, style]
            for stop in stopsToRemove:
                stop.parentNode.removeChild(stop)
                num += 1
                _num_elements_removed += 1
    return num