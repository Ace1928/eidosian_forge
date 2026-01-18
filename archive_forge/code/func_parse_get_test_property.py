from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_get_test_property(ctx, tokens, breakstack):
    """
  ::

    get_test_property(test property VAR)

  :see: https://cmake.org/cmake/help/latest/command/get_test_property.html
  """
    return StandardArgTree.parse(ctx, tokens, 3, {}, [], breakstack)