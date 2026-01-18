from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.parse.util import (
from cmakelang.parse.common import NodeType, TreeNode
@classmethod
def consume_trailing(cls, ctx, tokens, parent):
    """
    Consume sequential comment lines, removing tokens from the input list and
    appending the resulting node as a child to the provided parent
    """
    if not next_is_trailing_comment(ctx.config, tokens):
        return None
    if next_is_explicit_trailing_comment(ctx.config, tokens):
        return cls.consume_explicit_trailing(ctx, tokens, parent)
    return cls.consume_implicit_trailing(ctx, tokens, parent)