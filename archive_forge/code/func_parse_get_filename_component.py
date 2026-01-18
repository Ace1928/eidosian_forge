from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_get_filename_component(ctx, tokens, breakstack):
    """
  ::

      get_filename_component(<VAR> <FileName> <COMP> [CACHE])

      get_filename_component(<VAR> FileName
                             PROGRAM [PROGRAM_ARGS <ARG_VAR>]
                             [CACHE])

  :see: https://cmake.org/cmake/help/latest/command/get_filename_component.html
  """
    descriminator = get_nth_semantic_token(tokens, 2)
    if descriminator is not None and descriminator.spelling.upper() == 'PROGRAM':
        flags = ['PROGRAM', 'PROGRAM_ARGS', 'CACHE']
        return StandardArgTree.parse(ctx, tokens, '3+', {}, flags, breakstack)
    flags = ['DIRECTORY', 'NAME', 'EXT', 'NAME_WE', 'ABSOLUTE', 'REALPATH', 'PATH']
    return StandardArgTree.parse(ctx, tokens, '3+', {}, flags, breakstack)