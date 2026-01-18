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
def check_tokens(self, tokens):
    """Look for anything that looks like an incomplete variable substitution."""
    _, local_ctx = self.context
    missing_suffix = re.compile('(?<!\\\\)(?:\\\\\\\\)*(\\$\\{)((?:(?:[A-Za-z0-9_./+-])|(?:\\\\[^A-Za-z0-9_./+-]))+)')
    missing_prefix = re.compile('(?<!\\\\)(?:\\\\\\\\)*(\\$|\\{)((?:(?:[A-Za-z0-9_./+-])|(?:\\\\[^A-Za-z0-9_./+-]))+)')
    match_types = (TokenType.QUOTED_LITERAL, TokenType.UNQUOTED_LITERAL, TokenType.BRACKET_ARGUMENT)
    for token in tokens:
        if token.type not in match_types:
            continue
        resolved = mock_varrefs(token.spelling)
        match = missing_prefix.search(resolved)
        if match and variables.CASE_SENSITIVE_REGEX.match(match.group(2)):
            catmatch = ''.join(match.group(1, 2))
            if catmatch == '$ENV':
                continue
            local_ctx.record_lint('W0106', 'open', catmatch, location=token.get_location())
            continue
        match = missing_suffix.search(resolved)
        if match and variables.CASE_SENSITIVE_REGEX.match(match.group(2)):
            local_ctx.record_lint('W0106', 'closing', catmatch, location=token.get_location())
            continue