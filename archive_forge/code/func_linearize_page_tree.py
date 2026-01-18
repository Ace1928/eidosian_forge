from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
def linearize_page_tree(self, node=None):
    if node is None:
        node = self.page_tree_root
    check_format_condition(node[b'Type'] == b'Pages', '/Type of page tree node is not /Pages')
    pages = []
    for kid in node[b'Kids']:
        kid_object = self.read_indirect(kid)
        if kid_object[b'Type'] == b'Page':
            pages.append(kid)
        else:
            pages.extend(self.linearize_page_tree(node=kid_object))
    return pages