from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import six
def create_scope(self, node):
    subscope = Scope(self, node)
    self.get_root_scope()._set_scope_for_node(node, subscope)
    return subscope