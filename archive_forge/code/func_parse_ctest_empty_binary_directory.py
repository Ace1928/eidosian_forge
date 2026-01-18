from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_ctest_empty_binary_directory(ctx, tokens, breakstack):
    """
  ::

    ctest_empty_binary_directory( directory )

  :see: https://cmake.org/cmake/help/latest/command/ctest_empty_binary_directory.html
  """
    return StandardArgTree.parse(ctx, tokens, 1, {}, [], breakstack)