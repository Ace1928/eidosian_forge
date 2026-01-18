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
def removeUnreferencedElements(doc, keepDefs):
    """
    Removes all unreferenced elements except for <svg>, <font>, <metadata>, <title>, and <desc>.
    Also vacuums the defs of any non-referenced renderable elements.

    Returns the number of unreferenced elements removed from the document.
    """
    global _num_elements_removed
    num = 0
    removeTags = ['linearGradient', 'radialGradient', 'pattern']
    identifiedElements = findElementsWithId(doc.documentElement)
    referencedIDs = findReferencedElements(doc.documentElement)
    if not keepDefs:
        defs = doc.documentElement.getElementsByTagName('defs')
        for aDef in defs:
            elemsToRemove = removeUnusedDefs(doc, aDef, referencedIDs=referencedIDs)
            for elem in elemsToRemove:
                elem.parentNode.removeChild(elem)
                _num_elements_removed += 1
                num += 1
    for id in identifiedElements:
        if id not in referencedIDs:
            goner = identifiedElements[id]
            if goner is not None and goner.nodeName in removeTags and (goner.parentNode is not None) and (goner.parentNode.tagName != 'defs'):
                goner.parentNode.removeChild(goner)
                num += 1
                _num_elements_removed += 1
    return num