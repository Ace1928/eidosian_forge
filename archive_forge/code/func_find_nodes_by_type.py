from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import re
from pasta.augment import errors
from pasta.base import formatting as fmt
def find_nodes_by_type(node, accept_types):
    visitor = FindNodeVisitor(lambda n: isinstance(n, accept_types))
    visitor.visit(node)
    return visitor.results