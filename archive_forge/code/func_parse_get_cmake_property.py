from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_get_cmake_property(ctx, tokens, breakstack):
    """
  ::

      get_cmake_property(VAR property)

  :see: https://cmake.org/cmake/help/latest/command/get_cmake_property.html
  """
    return StandardArgTree.parse(ctx, tokens, 2, {}, [], breakstack)