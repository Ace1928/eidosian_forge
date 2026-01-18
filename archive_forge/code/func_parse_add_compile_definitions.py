from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_add_compile_definitions(ctx, tokens, breakstack):
    """
  ::

  add_compile_definitions(<definition> ...)

  :see: https://cmake.org/cmake/help/latest/command/add_compile_definitions.html
  """
    return StandardArgTree.parse(ctx, tokens, '+', {}, [], breakstack)