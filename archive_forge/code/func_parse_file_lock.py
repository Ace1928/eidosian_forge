import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import PatternNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import get_first_semantic_token
from cmakelang.parse.simple_nodes import consume_whitespace_and_comments
def parse_file_lock(ctx, tokens, breakstack):
    """
  ::

    file(LOCK <path> [DIRECTORY] [RELEASE]
         [GUARD <FUNCTION|FILE|PROCESS>]
         [RESULT_VARIABLE <variable>]
         [TIMEOUT <seconds>])

  :see: https://cmake.org/cmake/help/v3.14/command/file.html#locking
  """
    return StandardArgTree.parse(ctx, tokens, npargs='+', kwargs={'GUARD': PositionalParser(1, flags=['FUNCTION', 'FILE', 'PROCESS']), 'RESULT_VARIABLE': PositionalParser(1), 'TIMEOUT': PositionalParser(1)}, flags=['LOCK', 'DIRECTORY', 'RELEASE'], breakstack=breakstack)