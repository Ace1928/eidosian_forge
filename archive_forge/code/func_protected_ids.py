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
def protected_ids(seenIDs, options):
    """Return a list of protected IDs out of the seenIDs"""
    protectedIDs = []
    if options.protect_ids_prefix or options.protect_ids_noninkscape or options.protect_ids_list:
        protect_ids_prefixes = []
        protect_ids_list = []
        if options.protect_ids_list:
            protect_ids_list = options.protect_ids_list.split(',')
        if options.protect_ids_prefix:
            protect_ids_prefixes = options.protect_ids_prefix.split(',')
        for id in seenIDs:
            protected = False
            if options.protect_ids_noninkscape and (not id[-1].isdigit()):
                protected = True
            elif protect_ids_list and id in protect_ids_list:
                protected = True
            elif protect_ids_prefixes:
                if any((id.startswith(prefix) for prefix in protect_ids_prefixes)):
                    protected = True
            if protected:
                protectedIDs.append(id)
    return protectedIDs