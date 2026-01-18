import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import KwargBreaker, NodeType, TreeNode
from cmakelang.parse.util import (
def parse_add_subdirectory(ctx, tokens, breakstack):
    """
  ::

    add_subdirectory(source_dir [binary_dir] [EXCLUDE_FROM_ALL])

  :see: https://cmake.org/cmake/help/latest/command/add_subdirectory.html
  """
    return StandardArgTree.parse(ctx, tokens, '+', {}, ['EXCLUDE_FROM_ALL'], breakstack)