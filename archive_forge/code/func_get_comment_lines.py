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
def get_comment_lines(config, node):
    """Given a comment node, iterate through it's tokens and generate a list
  of textual lines."""
    inlines = []
    for token in node.children:
        assert isinstance(token, lex.Token), 'Unexpected object as child of comment node'
        if token.type == TokenType.COMMENT:
            inline = token.spelling.strip()
            if isinstance(node, simple_nodes.CommentNode) and node.is_explicit_trailing:
                inlines.append(inline[len(config.markup.explicit_trailing_pattern):])
                continue
            if not config.markup.enable_markup:
                inlines.append(inline[1:])
            elif inline.startswith('#' * config.markup.hashruler_min_length):
                inlines.append(inline[1:])
            else:
                inlines.append(inline.lstrip('#'))
    return inlines