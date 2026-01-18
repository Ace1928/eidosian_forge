from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_set_source_files_properties(ctx, tokens, breakstack):
    """
  ::

    set_source_files_properties([file1 [file2 [...]]]
                                PROPERTIES prop1 value1
                                [prop2 value2 [...]])

  :see: https://cmake.org/cmake/help/latest/command/set_source_files_properties.html
  """
    kwargs = {'PROPERTIES': TupleParser(2, '+')}
    return StandardArgTree.parse(ctx, tokens, '+', kwargs, [], breakstack)