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
def mergeSiblingGroupsWithCommonAttributes(elem):
    """
    Merge two or more sibling <g> elements with the identical attributes.

    This function acts recursively on the given element.
    """
    num = 0
    i = elem.childNodes.length - 1
    while i >= 0:
        currentNode = elem.childNodes.item(i)
        if currentNode.nodeType != Node.ELEMENT_NODE or currentNode.nodeName != 'g' or currentNode.namespaceURI != NS['SVG']:
            i -= 1
            continue
        attributes = {a.nodeName: a.nodeValue for a in currentNode.attributes.values()}
        if not attributes:
            i -= 1
            continue
        runStart, runEnd = (i, i)
        runElements = 1
        while runStart > 0:
            nextNode = elem.childNodes.item(runStart - 1)
            if nextNode.nodeType == Node.ELEMENT_NODE:
                if nextNode.nodeName != 'g' or nextNode.namespaceURI != NS['SVG']:
                    break
                nextAttributes = {a.nodeName: a.nodeValue for a in nextNode.attributes.values()}
                hasNoMergeTags = (True for n in nextNode.childNodes if n.nodeType == Node.ELEMENT_NODE and n.nodeName in ('title', 'desc') and (n.namespaceURI == NS['SVG']))
                if attributes != nextAttributes or any(hasNoMergeTags):
                    break
                else:
                    runElements += 1
                    runStart -= 1
            else:
                runStart -= 1
        i = runStart - 1
        if runElements < 2:
            continue
        while True:
            node = elem.childNodes.item(runStart)
            if node.nodeType == Node.ELEMENT_NODE and node.nodeName == 'g' and (node.namespaceURI == NS['SVG']):
                break
            runStart += 1
        primaryGroup = elem.childNodes.item(runStart)
        runStart += 1
        nodes = elem.childNodes[runStart:runEnd + 1]
        for node in nodes:
            if node.nodeType == Node.ELEMENT_NODE and node.nodeName == 'g' and (node.namespaceURI == NS['SVG']):
                for child in node.childNodes[:]:
                    primaryGroup.appendChild(child)
                elem.removeChild(node).unlink()
            else:
                primaryGroup.appendChild(node)
    for childNode in elem.childNodes:
        if childNode.nodeType == Node.ELEMENT_NODE:
            num += mergeSiblingGroupsWithCommonAttributes(childNode)
    return num