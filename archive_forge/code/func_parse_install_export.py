import logging
from cmakelang import lex
from cmakelang.parse.common import KwargBreaker
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import PatternNode
from cmakelang.parse.util import (
def parse_install_export(ctx, tokens, breakstack):
    """
  ::

    install(EXPORT <export-name> DESTINATION <dir>
            [NAMESPACE <namespace>] [[FILE <name>.cmake]|
            [PERMISSIONS permissions...]
            [CONFIGURATIONS [Debug|Release|...]]
            [EXPORT_LINK_INTERFACE_LIBRARIES]
            [COMPONENT <component>]
            [EXCLUDE_FROM_ALL])

  :see: https://cmake.org/cmake/help/v3.14/command/install.html#installing-exports
  """
    return StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={'EXPORT': PositionalParser(1), 'DESTINATION': PositionalParser(1), 'NAMESPACE': PositionalParser(1), 'FILE': PositionalParser(1), 'PERMISSIONS': PositionalParser('+'), 'CONFIGURATIONS': PositionalParser('+'), 'COMPONENT': PositionalParser(1)}, flags=['EXCLUDE_FROM_ALL'], breakstack=breakstack)