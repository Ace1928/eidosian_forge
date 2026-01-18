from __future__ import absolute_import
import re
import operator
import sys
def iter_recursive(node):
    for name in node.child_attrs:
        for child in iterchildren(node, name):
            if type_name(child) == node_name:
                yield child
            for c in iter_recursive(child):
                yield c