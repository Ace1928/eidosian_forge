import copy
import sys
import re
import os
from itertools import chain
from contextlib import contextmanager
from parso.python import tree
def deep_ast_copy(obj):
    """
    Much, much faster than copy.deepcopy, but just for parser tree nodes.
    """
    new_obj = copy.copy(obj)
    new_children = []
    for child in obj.children:
        if isinstance(child, tree.Leaf):
            new_child = copy.copy(child)
            new_child.parent = new_obj
        else:
            new_child = deep_ast_copy(child)
            new_child.parent = new_obj
        new_children.append(new_child)
    new_obj.children = new_children
    return new_obj