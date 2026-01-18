from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_utility_source(ctx, tokens, breakstack):
    """
  ::

    utility_source(cache_entry executable_name
                   path_to_source [file1 file2 ...])

  :see: https://cmake.org/cmake/help/latest/command/utility_source.html
  """
    return StandardArgTree.parse(ctx, tokens, '3+', {}, [], breakstack)