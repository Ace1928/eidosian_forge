from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_qt_wrap_cpp(ctx, tokens, breakstack):
    """
  ::

    qt_wrap_cpp(resultingLibraryName DestName SourceLists ...)

  :see: https://cmake.org/cmake/help/latest/command/qt_wrap_cpp.html
  """
    return StandardArgTree.parse(ctx, tokens, '3+', {}, [], breakstack)