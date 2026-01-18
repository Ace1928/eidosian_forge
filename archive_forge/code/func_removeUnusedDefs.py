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
def removeUnusedDefs(doc, defElem, elemsToRemove=None, referencedIDs=None):
    if elemsToRemove is None:
        elemsToRemove = []
    if referencedIDs is None:
        referencedIDs = findReferencedElements(doc.documentElement)
    keepTags = ['font', 'style', 'metadata', 'script', 'title', 'desc']
    for elem in defElem.childNodes:
        if elem.nodeType != Node.ELEMENT_NODE:
            continue
        elem_id = elem.getAttribute('id')
        if elem_id == '' or elem_id not in referencedIDs:
            if elem.nodeName == 'g' and elem.namespaceURI == NS['SVG']:
                elemsToRemove = removeUnusedDefs(doc, elem, elemsToRemove, referencedIDs=referencedIDs)
            elif elem.nodeName not in keepTags:
                elemsToRemove.append(elem)
    return elemsToRemove