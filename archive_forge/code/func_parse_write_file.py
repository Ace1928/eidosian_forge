from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_write_file(ctx, tokens, breakstack):
    """
  ::

    write_file(filename "message to write"... [APPEND])

  :see: https://cmake.org/cmake/help/latest/command/write_file.html
  """
    return StandardArgTree.parse(ctx, tokens, '2+', {}, ['APPEND'], breakstack)