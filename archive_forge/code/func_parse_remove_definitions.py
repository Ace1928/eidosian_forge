from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_remove_definitions(ctx, tokens, breakstack):
    """
  ::

    remove_definitions(-DFOO -DBAR ...)

  :see: https://cmake.org/cmake/help/latest/command/remove_definitions.html
  """
    return StandardArgTree.parse(ctx, tokens, '+', {}, [], breakstack)