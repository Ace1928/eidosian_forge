import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import KwargBreaker, NodeType, TreeNode
from cmakelang.parse.util import (
def parse_add_dependencies(ctx, tokens, breakstack):
    """
  ::

    add_dependencies(<target> [<target-dependency>]...)

  :see: https://cmake.org/cmake/help/latest/command/add_dependencies.html
  """
    return StandardArgTree.parse(ctx, tokens, '2+', {}, [], breakstack)