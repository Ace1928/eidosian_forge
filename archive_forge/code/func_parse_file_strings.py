import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import PatternNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import get_first_semantic_token
from cmakelang.parse.simple_nodes import consume_whitespace_and_comments
def parse_file_strings(ctx, tokens, breakstack):
    """
  ::

    file(STRINGS <filename> <variable> [<options>...])

  :see: https://cmake.org/cmake/help/v3.14/command/file.html#read
  """
    return StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={'LENGTH_MAXIMUM': PositionalParser(1), 'LENGTH_MINIMUM': PositionalParser(1), 'LIMIT_COUNT': PositionalParser(1), 'LIMIT_INPUT': PositionalParser(1), 'LIMIT_OUTPUT': PositionalParser(1), 'REGEX': PositionalParser(1), 'ENCODING': PositionalParser(1, flags=['UTF-8', 'UTF-16LE', 'UTF-16BE', 'UTF-32LE', 'UTF-32BE'])}, flags=['STRINGS', 'NEWLINE_CONSUME', 'NO_HEX_CONVERSION'], breakstack=breakstack)