import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import PatternNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import get_first_semantic_token
from cmakelang.parse.simple_nodes import consume_whitespace_and_comments
def parse_file_generate_output(ctx, tokens, breakstack):
    """
  ::

    file(GENERATE OUTPUT output-file
        <INPUT input-file|CONTENT content>
        [CONDITION expression])

  :see: https://cmake.org/cmake/help/v3.14/command/file.html#writing
  """
    return StandardArgTree.parse(ctx, tokens, npargs=1, kwargs={'OUTPUT': PositionalParser(1), 'INPUT': PositionalParser(1), 'CONTENT': PositionalParser('+'), 'CONDITION': ConditionalGroupNode.parse}, flags=['GENERATE'], breakstack=breakstack)