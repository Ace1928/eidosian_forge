import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import PatternNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import get_first_semantic_token
from cmakelang.parse.simple_nodes import consume_whitespace_and_comments
def parse_file_create_link(ctx, tokens, breakstack):
    """
  ::

    file(CREATE_LINK <original> <linkname>
        [RESULT <result>] [COPY_ON_ERROR] [SYMBOLIC])

  :see: https://cmake.org/cmake/help/v3.14/command/file.html#filesystem
  """
    return StandardArgTree.parse(ctx, tokens, npargs='+', kwargs={'RESULT': PositionalParser(1)}, flags=['COPY_ON_ERROR', 'SYMBOLIC'], breakstack=breakstack)