import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import PatternNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import get_first_semantic_token
from cmakelang.parse.simple_nodes import consume_whitespace_and_comments
def parse_file_glob(ctx, tokens, breakstack):
    """
  ::

    file(GLOB <variable>
        [LIST_DIRECTORIES true|false] [RELATIVE <path>] [CONFIGURE_DEPENDS]
        [<globbing-expressions>...])
    file(GLOB_RECURSE <variable> [FOLLOW_SYMLINKS]
        [LIST_DIRECTORIES true|false] [RELATIVE <path>] [CONFIGURE_DEPENDS]
        [<globbing-expressions>...])
  :see: https://cmake.org/cmake/help/v3.14/command/file.html#filesystem
  """
    return StandardArgTree.parse(ctx, tokens, npargs='+', kwargs={'LIST_DIRECTORIES': PositionalParser(1), 'RELATIVE': PositionalParser(1)}, flags=['GLOB', 'GLOB_RECURSE', 'CONFIGURE_DEPENDS', 'FOLLOW_SYMLINKS'], breakstack=breakstack)