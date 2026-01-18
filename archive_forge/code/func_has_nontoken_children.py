from __future__ import print_function
from __future__ import unicode_literals
import io
import sys
from cmakelang.parse.common import TreeNode
def has_nontoken_children(node):
    if not hasattr(node, 'children'):
        return False
    for child in node.children:
        if isinstance(child, TreeNode):
            return True
    return False