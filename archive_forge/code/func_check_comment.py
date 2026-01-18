import enum
import logging
import re
from cmakelang.common import InternalError
from cmakelang.format.formatter import get_comment_lines
from cmakelang.lex import TokenType, Token
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.body_nodes import BodyNode, FlowControlNode
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.common import NodeType, TreeNode
from cmakelang.parse.statement_node import StatementNode
from cmakelang.parse.util import get_min_npargs
from cmakelang.parse import variables
from cmakelang.parse.funs.set import SetFnNode
def check_comment(self, node):
    if not self.am_in_statement():
        return
    requested_suppressions = self.parse_pragmas_from_comment(node)
    if not requested_suppressions:
        return
    _cfg, local_ctx = self.context
    lineno = node.get_tokens()[0].get_location().line
    new_suppressions = local_ctx.suppress(lineno, requested_suppressions)
    local_ctx.unsuppress(lineno + 1, new_suppressions)