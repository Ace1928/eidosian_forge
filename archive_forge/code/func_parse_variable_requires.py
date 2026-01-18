from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_variable_requires(ctx, tokens, breakstack):
    """
  ::

    variable_requires(TEST_VARIABLE RESULT_VARIABLE
                      REQUIRED_VARIABLE1
                      REQUIRED_VARIABLE2 ...)

  :see: https://cmake.org/cmake/help/latest/command/variable_requires.html
  """
    return StandardArgTree.parse(ctx, tokens, '3+', {}, [], breakstack)