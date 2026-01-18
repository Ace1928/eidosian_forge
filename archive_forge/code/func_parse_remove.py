from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_remove(ctx, tokens, breakstack):
    """
  ::

    remove(VAR VALUE VALUE ...)

  :see: https://cmake.org/cmake/help/latest/command/remove.html
  """
    return StandardArgTree.parse(ctx, tokens, '2+', {}, [], breakstack)