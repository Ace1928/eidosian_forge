import logging
from cmakelang import lex
from cmakelang.parse.common import KwargBreaker, NodeType
from cmakelang.parse.common import TreeNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_foreach_standard(ctx, tokens, breakstack):
    """
  ::

    foreach(<loop_var> <items>)
      <commands>
    endforeach()
  """
    tree = ArgGroupNode()
    while tokens and tokens[0].type in WHITESPACE_TOKENS:
        tree.children.append(tokens.pop(0))
        continue
    ntokens = len(tokens)
    subtree = PositionalGroupNode.parse(ctx, tokens, 1, [], breakstack)
    assert len(tokens) < ntokens
    tree.children.append(subtree)
    ntokens = len(tokens)
    subtree = PositionalGroupNode.parse(ctx, tokens, '+', [], breakstack)
    assert len(tokens) < ntokens
    tree.children.append(subtree)
    return tree