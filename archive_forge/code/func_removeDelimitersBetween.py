from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
@staticmethod
def removeDelimitersBetween(bottom, top):
    if bottom.get('next') != top:
        bottom['next'] = top
        top['previous'] = bottom