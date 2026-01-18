from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_ctest_start(ctx, tokens, breakstack):
    """
  ::

    ctest_start(<model> [<source> [<binary>]] [GROUP <group>] [QUIET])
    ctest_start([<model> [<source> [<binary>]]] [GROUP <group>] APPEND [QUIET])

  :see: https://cmake.org/cmake/help/latest/command/ctest_start.html
  """
    kwargs = {'GROUP': PositionalParser(1)}
    flags = ['QUIET', 'APPEND']
    return StandardArgTree.parse(ctx, tokens, '+', kwargs, flags, breakstack)