from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_variable_watch(ctx, tokens, breakstack):
    """
  ::

    variable_watch(<variable> [<command>])

  :see: https://cmake.org/cmake/help/latest/command/variable_watch.html
  """
    return StandardArgTree.parse(ctx, tokens, '+', {}, [], breakstack)