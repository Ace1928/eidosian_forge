import logging
from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import NodeType, TreeNode
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.util import (
def parse_add_executable_imported(ctx, tokens, breakstack):
    """
  ::

    add_executable(<name> IMPORTED [GLOBAL])

  :see: https://cmake.org/cmake/help/latest/command/add_executable.html
  """
    return StandardArgTree.parse(ctx, tokens, npargs='+', kwargs={}, flags=['IMPORTED', 'GLOBAL'], breakstack=breakstack)