from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_make_directory(ctx, tokens, breakstack):
    """
  ::

    make_directory(directory)

  :see: https://cmake.org/cmake/help/latest/command/make_directory.html
  """
    return StandardArgTree.parse(ctx, tokens, 1, {}, [], breakstack)