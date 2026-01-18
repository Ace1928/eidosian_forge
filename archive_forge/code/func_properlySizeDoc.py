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
def properlySizeDoc(docElement, options):
    w = SVGLength(docElement.getAttribute('width'))
    h = SVGLength(docElement.getAttribute('height'))
    if options.renderer_workaround:
        if w.units != Unit.NONE and w.units != Unit.PX or (h.units != Unit.NONE and h.units != Unit.PX):
            return
    vbSep = RE_COMMA_WSP.split(docElement.getAttribute('viewBox'))
    vbWidth, vbHeight = (0, 0)
    if len(vbSep) == 4:
        try:
            vbX = float(vbSep[0])
            vbY = float(vbSep[1])
            if vbX != 0 or vbY != 0:
                return
            vbWidth = float(vbSep[2])
            vbHeight = float(vbSep[3])
            if vbWidth != w.value or vbHeight != h.value:
                return
        except ValueError:
            pass
    docElement.setAttribute('viewBox', '0 0 %s %s' % (w.value, h.value))
    docElement.removeAttribute('width')
    docElement.removeAttribute('height')