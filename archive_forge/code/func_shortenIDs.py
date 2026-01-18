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
def shortenIDs(doc, prefix, options):
    """
    Shortens ID names used in the document. ID names referenced the most often are assigned the
    shortest ID names.

    Returns the number of bytes saved by shortening ID names in the document.
    """
    num = 0
    identifiedElements = findElementsWithId(doc.documentElement)
    referencedIDs = findReferencedElements(doc.documentElement)
    idList = [(len(referencedIDs[rid]), rid) for rid in referencedIDs if rid in identifiedElements]
    idList.sort(reverse=True)
    idList = [rid for count, rid in idList]
    idList.extend([rid for rid in identifiedElements if rid not in idList])
    protectedIDs = protected_ids(identifiedElements, options)
    consumedIDs = set()
    need_new_id = []
    id_allocations = list(compute_id_lengths(len(idList) + 1))
    id_allocations.reverse()
    optimal_id_length, id_use_limit = (0, 0)
    for current_id in idList:
        if id_use_limit < 1:
            optimal_id_length, id_use_limit = id_allocations.pop()
        id_use_limit -= 1
        if len(current_id) == optimal_id_length:
            consumedIDs.add(current_id)
        else:
            need_new_id.append(current_id)
    curIdNum = 1
    for old_id in need_new_id:
        new_id = intToID(curIdNum, prefix)
        while new_id in protectedIDs or new_id in consumedIDs:
            curIdNum += 1
            new_id = intToID(curIdNum, prefix)
        num += renameID(old_id, new_id, identifiedElements, referencedIDs.get(old_id))
        curIdNum += 1
    return num