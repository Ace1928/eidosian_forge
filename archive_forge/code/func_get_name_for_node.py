from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import six
def get_name_for_node(self, node):
    return self._nodes_to_names.get(node, None)