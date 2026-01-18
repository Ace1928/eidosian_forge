from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import (
from cmakelang.parse.util import (
def parse_empty(ctx, tokens, breakstack):
    """
  ::

    break()
    continue()
    enable_testing()

  :see: https://cmake.org/cmake/help/latest/command/break.html
  :see: https://cmake.org/cmake/help/latest/command/continue.html
  :see: https://cmake.org/cmake/help/latest/command/enable_testing.html
  :see: https://cmake.org/cmake/help/latest/command/return.html
  """
    tree = ArgGroupNode()
    while tokens:
        if should_break(tokens[0], breakstack):
            break
        if tokens[0].type in WHITESPACE_TOKENS:
            tree.children.append(tokens.pop(0))
            continue
        if tokens[0].type in (lex.TokenType.COMMENT, lex.TokenType.BRACKET_COMMENT):
            child = TreeNode(NodeType.COMMENT)
            tree.children.append(child)
            child.children.append(tokens.pop(0))
            continue
        ctx.lint_ctx.record_lint('E1121', location=tokens[0].get_location())
        ntokens = len(tokens)
        subtree = PositionalGroupNode.parse(ctx, tokens, '+', [], breakstack)
        assert len(tokens) < ntokens
        tree.children.append(subtree)
    return tree