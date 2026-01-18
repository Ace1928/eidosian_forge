import logging
from cmakelang import lex
from cmakelang.parse.common import KwargBreaker
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import PatternNode
from cmakelang.parse.util import (
def parse_install_directory(ctx, tokens, breakstack):
    """
  ::

    install(DIRECTORY dirs...
            TYPE <type> | DESTINATION <dir>
            [FILE_PERMISSIONS permissions...]
            [DIRECTORY_PERMISSIONS permissions...]
            [USE_SOURCE_PERMISSIONS] [OPTIONAL] [MESSAGE_NEVER]
            [CONFIGURATIONS [Debug|Release|...]]
            [COMPONENT <component>] [EXCLUDE_FROM_ALL]
            [FILES_MATCHING]
            [[PATTERN <pattern> | REGEX <regex>]
             [EXCLUDE] [PERMISSIONS permissions...]] [...])

  :see: https://cmake.org/cmake/help/v3.14/command/install.html#directory
  """
    return StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={'DIRECTORY': PositionalParser('+'), 'TYPE': PositionalParser(1), 'DESTINATION': PositionalParser(1), 'FILE_PERMISSIONS': PositionalParser('+'), 'DIRECTORY_PERMISSIONS': PositionalParser('+'), 'CONFIGURATIONS': PositionalParser('+'), 'COMPONENT': PositionalParser(1), 'RENAME': PositionalParser(1), 'PATTERN': PatternNode.parse, 'REGEX': PatternNode.parse}, flags=['USER_SOURCE_PERMISSIONS', 'OPTIONAL', 'MESSAGE_NEVER', 'FILES_MATCHING'], breakstack=breakstack)