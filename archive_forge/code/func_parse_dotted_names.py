import copy
import sys
import re
import os
from itertools import chain
from contextlib import contextmanager
from parso.python import tree
def parse_dotted_names(nodes, is_import_from, until_node=None):
    level = 0
    names = []
    for node in nodes[1:]:
        if node in ('.', '...'):
            if not names:
                level += len(node.value)
        elif node.type == 'dotted_name':
            for n in node.children[::2]:
                names.append(n)
                if n is until_node:
                    break
            else:
                continue
            break
        elif node.type == 'name':
            names.append(node)
            if node is until_node:
                break
        elif node == ',':
            if not is_import_from:
                names = []
        else:
            break
    return (level, names)