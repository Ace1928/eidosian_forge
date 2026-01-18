import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import PatternNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import get_first_semantic_token
from cmakelang.parse.simple_nodes import consume_whitespace_and_comments
def parse_file_hash(ctx, tokens, breakstack):
    """
  ::

    file(<HASH> <filename> <variable>)

  :see: https://cmake.org/cmake/help/v3.14/command/file.html#strings
  """
    return StandardArgTree.parse(ctx, tokens, npargs=3, kwargs={}, flags=HASH_STRINGS, breakstack=breakstack)