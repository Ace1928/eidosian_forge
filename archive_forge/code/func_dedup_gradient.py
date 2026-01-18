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
def dedup_gradient(master_id, duplicates_ids, duplicates, referenced_ids):
    func_iri = None
    for dup_id, dup_grad in zip(duplicates_ids, duplicates):
        if not dup_grad.parentNode:
            continue
        if dup_id in referenced_ids:
            if func_iri is None:
                dup_id_regex = '|'.join(duplicates_ids)
                func_iri = re.compile('url\\([\'"]?#(?:' + dup_id_regex + ')[\'"]?\\)')
            for elem in referenced_ids[dup_id]:
                for attr in ['fill', 'stroke']:
                    v = elem.getAttribute(attr)
                    v_new, n = func_iri.subn('url(#' + master_id + ')', v)
                    if n > 0:
                        elem.setAttribute(attr, v_new)
                if elem.getAttributeNS(NS['XLINK'], 'href') == '#' + dup_id:
                    elem.setAttributeNS(NS['XLINK'], 'href', '#' + master_id)
                styles = _getStyle(elem)
                for style in styles:
                    v = styles[style]
                    v_new, n = func_iri.subn('url(#' + master_id + ')', v)
                    if n > 0:
                        styles[style] = v_new
                _setStyle(elem, styles)
        dup_grad.parentNode.removeChild(dup_grad)
    if master_id:
        try:
            master_references = referenced_ids[master_id]
        except KeyError:
            master_references = set()
        for dup_id in duplicates_ids:
            references = referenced_ids.pop(dup_id, None)
            if references is None:
                continue
            master_references.update(references)
        referenced_ids[master_id] = master_references