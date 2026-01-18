import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import PatternNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import get_first_semantic_token
from cmakelang.parse.simple_nodes import consume_whitespace_and_comments
def parse_file_timestamp(ctx, tokens, breakstack):
    """
  ::

    file(TIMESTAMP <filename> <variable> [<format>] [UTC])

  :see: https://cmake.org/cmake/help/v3.14/command/file.html#strings
  """
    return StandardArgTree.parse(ctx, tokens, npargs='+', kwargs={}, flags=['TIMESTAMP', 'UTC'], breakstack=breakstack)