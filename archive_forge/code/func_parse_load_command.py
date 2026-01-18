from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_load_command(ctx, tokens, breakstack):
    """
  ::

    load_command(COMMAND_NAME <loc1> [loc2 ...])

  :see: https://cmake.org/cmake/help/latest/command/load_command.html
  """
    return StandardArgTree.parse(ctx, tokens, '2+', {}, ['COMMAND_NAME'], breakstack)