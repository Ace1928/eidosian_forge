from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_create_test_sourcelist(ctx, tokens, breakstack):
    """
  ::

    create_test_sourcelist(sourceListName driverName
                           test1 test2 test3
                           EXTRA_INCLUDE include.h
                           FUNCTION function)

  :see: https://cmake.org/cmake/help/latest/command/create_test_sourcelist.html
  """
    kwargs = {'EXTRA_INCLUDE': PositionalParser(1), 'FUNCTION': PositionalParser(1)}
    return StandardArgTree.parse(ctx, tokens, '3+', kwargs, [], breakstack)