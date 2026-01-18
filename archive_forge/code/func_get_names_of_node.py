import copy
import sys
import re
import os
from itertools import chain
from contextlib import contextmanager
from parso.python import tree
def get_names_of_node(node):
    try:
        children = node.children
    except AttributeError:
        if node.type == 'name':
            return [node]
        else:
            return []
    else:
        return list(chain.from_iterable((get_names_of_node(c) for c in children)))