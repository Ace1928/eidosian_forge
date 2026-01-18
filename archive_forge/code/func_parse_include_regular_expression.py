from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_include_regular_expression(ctx, tokens, breakstack):
    """
  ::

    include_regular_expression(regex_match [regex_complain])

  :see: https://cmake.org/cmake/help/latest/command/include_regular_expression.html
  """
    return StandardArgTree.parse(ctx, tokens, '*', {}, [], breakstack)