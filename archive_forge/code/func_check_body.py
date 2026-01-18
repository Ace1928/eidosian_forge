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
def check_body(self, node):
    """Perform checks on a body node."""
    cfg, local_ctx = self.context
    suppressions = []
    prevchild = [None, None]
    for idx, child in enumerate(node.children):
        if not isinstance(child, TreeNode):
            continue
        if child.node_type is NodeType.COMMENT:
            requested_suppressions = self.parse_pragmas_from_comment(child)
            if requested_suppressions:
                lineno = child.get_tokens()[0].get_location().line
                new_suppressions = local_ctx.suppress(lineno, requested_suppressions)
                suppressions.extend(new_suppressions)
        if child.node_type is NodeType.FLOW_CONTROL:
            stmt = child.children[0]
            if statement_is_fundef(stmt):
                prefix_comment = get_prefix_comment(prevchild)
                if not prefix_comment:
                    local_ctx.record_lint('C0111', location=child.get_semantic_tokens()[0].get_location())
                elif not ''.join(get_comment_lines(cfg, prefix_comment)).strip():
                    local_ctx.record_lint('C0112', location=child.get_semantic_tokens()[0].get_location())
        if child.node_type in (NodeType.FLOW_CONTROL, NodeType.STATEMENT):
            if prevchild[0] is None:
                pass
            elif prevchild[0].node_type is not NodeType.WHITESPACE:
                local_ctx.record_lint('C0321', location=child.get_location())
            elif prevchild[0].count_newlines() < 1:
                local_ctx.record_lint('C0321', location=child.get_location())
            elif prevchild[0].count_newlines() < cfg.lint.min_statement_spacing:
                local_ctx.record_lint('C0305', 'not enough', location=child.get_location())
            elif prevchild[0].count_newlines() > cfg.lint.max_statement_spacing:
                local_ctx.record_lint('C0305', 'too many', location=child.get_location())
        if isinstance(child, StatementNode) and child.get_funname() in ('break', 'continue', 'return'):
            for _ in find_nodes_in_subtree(node.children[idx + 1:], StatementNode):
                local_ctx.record_lint('W0101', location=child.get_location())
                break
        prevchild[1] = prevchild[0]
        prevchild[0] = child
    lineno = node.get_tokens()[-1].get_location().line
    if suppressions:
        local_ctx.unsuppress(lineno, suppressions)