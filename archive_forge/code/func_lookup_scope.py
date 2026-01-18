from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import six
def lookup_scope(self, node):
    while node:
        try:
            return self._node_scopes[node]
        except KeyError:
            node = self.parent(node)
    return None