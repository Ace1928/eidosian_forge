from __future__ import absolute_import
import re
import operator
import sys
def find_first(node, path):
    return _get_first_or_none(iterfind(node, path))