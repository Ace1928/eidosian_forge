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
def collapseSinglyReferencedGradients(doc):
    global _num_elements_removed
    num = 0
    identifiedElements = findElementsWithId(doc.documentElement)
    for rid, nodes in six.iteritems(findReferencedElements(doc.documentElement)):
        if len(nodes) == 1 and rid in identifiedElements:
            elem = identifiedElements[rid]
            if elem is not None and elem.nodeType == Node.ELEMENT_NODE and (elem.nodeName in ['linearGradient', 'radialGradient']) and (elem.namespaceURI == NS['SVG']):
                refElem = nodes.pop()
                if refElem.nodeType == Node.ELEMENT_NODE and refElem.nodeName in ['linearGradient', 'radialGradient'] and (refElem.namespaceURI == NS['SVG']):
                    if len(refElem.getElementsByTagName('stop')) == 0:
                        stopsToAdd = elem.getElementsByTagName('stop')
                        for stop in stopsToAdd:
                            refElem.appendChild(stop)
                    for attr in ['gradientUnits', 'spreadMethod', 'gradientTransform']:
                        if refElem.getAttribute(attr) == '' and (not elem.getAttribute(attr) == ''):
                            refElem.setAttributeNS(None, attr, elem.getAttribute(attr))
                    if elem.nodeName == 'radialGradient' and refElem.nodeName == 'radialGradient':
                        for attr in ['fx', 'fy', 'cx', 'cy', 'r']:
                            if refElem.getAttribute(attr) == '' and (not elem.getAttribute(attr) == ''):
                                refElem.setAttributeNS(None, attr, elem.getAttribute(attr))
                    if elem.nodeName == 'linearGradient' and refElem.nodeName == 'linearGradient':
                        for attr in ['x1', 'y1', 'x2', 'y2']:
                            if refElem.getAttribute(attr) == '' and (not elem.getAttribute(attr) == ''):
                                refElem.setAttributeNS(None, attr, elem.getAttribute(attr))
                    target_href = elem.getAttributeNS(NS['XLINK'], 'href')
                    if target_href:
                        refElem.setAttributeNS(NS['XLINK'], 'href', target_href)
                    else:
                        refElem.removeAttributeNS(NS['XLINK'], 'href')
                    elem.parentNode.removeChild(elem)
                    _num_elements_removed += 1
                    num += 1
    return num