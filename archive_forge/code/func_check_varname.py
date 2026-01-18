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
def check_varname(self, varname, token, contextstr):
    """
    Record lint if the varname is a case-insensitive match to any builtin
    variable names, meaning that the author likely made a spelling mistake.
    """
    _, local_ctx = self.context
    imatch = variables.CASE_INSENSITIVE_REGEX.match(varname)
    if not imatch:
        return
    for groupstr in imatch.groups():
        if groupstr is not None:
            return
    if not variables.CASE_SENSITIVE_REGEX.match(varname):
        local_ctx.record_lint('W0105', contextstr, varname, location=token.get_location())