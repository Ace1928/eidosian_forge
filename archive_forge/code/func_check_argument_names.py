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
def check_argument_names(self, defn_node):
    """Check that the argument names in a function or macro definition match
      the required pattern."""
    cfg, local_ctx = self.context
    tokens = defn_node.argtree.get_semantic_tokens()[1:]
    seen_names = set()
    uncase_names = set()
    for token in tokens:
        if token.type is TokenType.RIGHT_PAREN:
            break
        if token.type is not TokenType.WORD:
            local_ctx.record_lint('E0109', token.spelling, location=token.get_location())
        else:
            if token.spelling in seen_names:
                local_ctx.record_lint('E0108', token.spelling, location=token.get_location())
            elif token.spelling.lower() in uncase_names:
                local_ctx.record_lint('C0202', token.spelling, location=token.get_location())
            elif not re.match(cfg.lint.argument_var_pattern, token.spelling):
                local_ctx.record_lint('C0103', 'argument', token.spelling, cfg.lint.argument_var_pattern, location=token.get_location())
            seen_names.add(token.spelling)
            uncase_names.add(token.spelling.lower())
    if len(tokens) > cfg.lint.max_arguments:
        local_ctx.record_lint('R0913', len(tokens), cfg.lint.max_arguments, location=defn_node.get_location())