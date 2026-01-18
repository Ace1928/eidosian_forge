from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_subdirs(ctx, tokens, breakstack):
    """
  ::

    subdirs(dir1 dir2 ...[EXCLUDE_FROM_ALL exclude_dir1 exclude_dir2 ...]
            [PREORDER] )

  :see: https://cmake.org/cmake/help/latest/command/subdirs.html
  """
    kwargs = {'EXCLUDE_FROM_ALL': PositionalParser('+')}
    flags = ['PREORDER']
    return StandardArgTree.parse(ctx, tokens, '+', kwargs, flags, breakstack)