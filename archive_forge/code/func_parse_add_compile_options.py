import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import KwargBreaker, NodeType, TreeNode
from cmakelang.parse.util import (
def parse_add_compile_options(ctx, tokens, breakstack):
    """
  ::

    add_compile_options(<option> ...)

  :see: https://cmake.org/cmake/help/latest/command/add_compile_options.html
  """
    return StandardArgTree.parse(ctx, tokens, '+', {}, [], breakstack)