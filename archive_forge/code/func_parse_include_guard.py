from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_include_guard(ctx, tokens, breakstack):
    """
  ::

    include_guard([DIRECTORY|GLOBAL])

  :see: https://cmake.org/cmake/help/latest/command/include_guard.html
  """
    return StandardArgTree.parse(ctx, tokens, '*', {}, [], breakstack)