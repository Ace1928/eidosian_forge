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
def check_assignment(self, node):
    """Checks on a variable assignment."""
    cfg, local_ctx = self.context
    scope = get_scope_of_assignment(node)
    if node.get_funname() == 'set':
        varname = node.argtree.varname
    elif node.get_funname() == 'list':
        varname = get_list_outvar(node)
        if varname is None:
            return
    else:
        logger.warning('Unexpected node %s (%s)', node, node.get_funname())
        return
    resolved = mock_varrefs(varname.spelling, '')
    if not resolved:
        return
    if scope is Scope.CACHE:
        pattern = cfg.lint.global_var_pattern
        if not re.match(pattern, resolved):
            local_ctx.record_lint('C0103', 'CACHE variable', varname.spelling, pattern, location=varname.get_location())
    elif scope is Scope.INTERNAL:
        pattern = cfg.lint.internal_var_pattern
        if not re.match(pattern, resolved):
            local_ctx.record_lint('C0103', 'INTERNAL variable', varname.spelling, cfg.lint.internal_var_pattern, location=varname.get_location())
    elif scope is Scope.PARENT:
        pattern = '|'.join([cfg.lint.public_var_pattern, cfg.lint.private_var_pattern, cfg.lint.local_var_pattern])
        if not re.match(pattern, resolved):
            local_ctx.record_lint('C0103', 'PARENT_SCOPE variable', varname.spelling, pattern, location=varname.get_location())
    elif scope is Scope.LOCAL:
        pattern = cfg.lint.local_var_pattern
        if not re.match(pattern, resolved):
            local_ctx.record_lint('C0103', 'local variable', varname.spelling, pattern, location=varname.get_location())
    elif scope is Scope.DIRECTORY:
        pattern = '|'.join([cfg.lint.public_var_pattern, cfg.lint.private_var_pattern])
        if not re.match(pattern, resolved):
            local_ctx.record_lint('C0103', 'directory variable', varname.spelling, pattern, location=varname.get_location())