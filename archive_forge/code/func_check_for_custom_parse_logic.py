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
def check_for_custom_parse_logic(self, stmt_node):
    """Ensure that a function or macro definition doesn't contain custom parser
      logic. The check is heuristic, but what we look for is a loop over ARGN
      where the body of the loop contains multiple conditional checks against
      the string value of the arguments
    """
    cfg, local_ctx = self.context
    block = stmt_node.parent.get_block_with(stmt_node)
    for _ in find_statements_in_subtree(block.body, ('cmake_parse_arguments',)):
        return
    for loop_stmt in find_statements_in_subtree(block.body, ('foreach', 'while')):
        if loop_contains_argn(loop_stmt):
            conditional_count = 0
            loopvar = loop_stmt.argtree.get_semantic_tokens()[0]
            loop_body = loop_stmt.parent.get_block_with(loop_stmt).body
            for conditional in find_nodes_in_subtree(loop_body, ConditionalGroupNode):
                tokens = conditional.get_semantic_tokens()
                if not tokens:
                    continue
                if tokens[0].spelling == loopvar.spelling:
                    if tokens[1].spelling in ('STREQUAL', 'MATCHES'):
                        conditional_count += 1
            if conditional_count > cfg.lint.max_conditionals_custom_parser:
                local_ctx.record_lint('C0201', location=loop_stmt.get_location())
                return