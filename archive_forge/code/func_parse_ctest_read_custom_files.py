from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_ctest_read_custom_files(ctx, tokens, breakstack):
    """
  ::

    ctest_read_custom_files( directory ... )

  :see: https://cmake.org/cmake/help/latest/command/ctest_read_custom_files.html
  """
    return StandardArgTree.parse(ctx, tokens, '+', {}, [], breakstack)