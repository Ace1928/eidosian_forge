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
@property
def colextent(self):
    """
    The column index of the right-most character in the layout of the
    subtree rooted at this node. In other words, the width of the
    bounding box for the subtree rooted at this node.
    """
    return self._colextent