import logging
from cmakelang import lex
from cmakelang.parse.common import KwargBreaker, NodeType
from cmakelang.parse.common import TreeNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_foreach_range(ctx, tokens, breakstack):
    """
  ::

    foreach(<loop_var> RANGE <start> <stop> [<step>])
      <commands>
    endforeach()
  """
    return StandardArgTree.parse(ctx, tokens, npargs=1, kwargs={'RANGE': PositionalParser(3)}, flags=[], breakstack=breakstack)