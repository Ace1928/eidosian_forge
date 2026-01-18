from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_qt_wrap_ui(ctx, tokens, breakstack):
    """
  ::

    qt_wrap_ui(resultingLibraryName HeadersDestName
               SourcesDestName SourceLists ...)

  :see: https://cmake.org/cmake/help/latest/command/qt_wrap_ui.html
  """
    return StandardArgTree.parse(ctx, tokens, '4+', {}, [], breakstack)