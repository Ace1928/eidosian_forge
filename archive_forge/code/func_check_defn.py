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
def check_defn(self, defn_node, name_pattern):
    """Perform checks on a function or macro"""
    cfg, local_ctx = self.context
    self.check_for_custom_parse_logic(defn_node)
    self.check_name_against_pattern(defn_node, name_pattern)
    self.check_argument_names(defn_node)
    body_line = defn_node.get_tokens()[-1].get_location().line + 1
    block = defn_node.parent.get_block_with(defn_node)
    return_count = sum((1 for _ in find_statements_in_subtree(block.body, ('return',))))
    if return_count > cfg.lint.max_returns:
        local_ctx.record_lint('R0911', return_count, cfg.lint.max_returns, location=(body_line,))
    branch_count = sum((1 for _ in find_statements_in_subtree(block.body, ('if', 'elseif', 'else'))))
    if branch_count > cfg.lint.max_branches:
        local_ctx.record_lint('R0912', branch_count, cfg.lint.max_branches, location=(body_line,))
    stmt_count = sum((1 for _ in find_nodes_in_subtree(block.body, StatementNode)))
    if stmt_count > cfg.lint.max_statements:
        local_ctx.record_lint('R0915', stmt_count, cfg.lint.max_statements, location=(body_line,))