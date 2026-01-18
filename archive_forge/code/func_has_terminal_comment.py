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
def has_terminal_comment(self):
    children = list(self.children)
    while children and children[0].node_type != NodeType.RPAREN:
        children.pop(0)
    if children:
        children.pop(0)
    return children and children[-1].pnode.children[0].type == TokenType.COMMENT