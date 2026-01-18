from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_site_name(ctx, tokens, breakstack):
    """
  ::

    site_name(variable)

  :see: https://cmake.org/cmake/help/latest/command/site_name.html
  """
    return StandardArgTree.parse(ctx, tokens, 1, {}, [], breakstack)