from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import io
import logging
import re
import sys
from cmakelang import lex
from cmakelang import markup
from cmakelang.common import UserError
from cmakelang.lex import TokenType
from cmakelang.parse.argument_nodes import PositionalGroupNode
from cmakelang.parse.common import FlowType, NodeType, TreeNode
from cmakelang.parse.util import comment_is_tag
from cmakelang.parse import simple_nodes
def create_box_tree(pnode):
    """
  Recursively construct a layout tree from the given parse tree
  """
    layout_root = LayoutNode.create(pnode)
    child_queue = list(pnode.children)
    while child_queue:
        pchild = child_queue.pop(0)
        if not isinstance(pchild, TreeNode):
            continue
        if pchild.node_type == NodeType.WHITESPACE and pchild.count_newlines() < 2:
            continue
        if pchild.node_type in MATCH_TYPES:
            layout_root.children.append(create_box_tree(pchild))
    return layout_root