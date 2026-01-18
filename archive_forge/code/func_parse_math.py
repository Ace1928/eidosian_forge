from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_math(ctx, tokens, breakstack):
    """
  ::

    math(EXPR <variable> "<expression>" [OUTPUT_FORMAT <format>])

  :see: https://cmake.org/cmake/help/latest/command/math.html
  """
    kwargs = {'OUTPUT_FORMAT': PositionalParser(1)}
    return StandardArgTree.parse(ctx, tokens, '3', kwargs, [], breakstack)