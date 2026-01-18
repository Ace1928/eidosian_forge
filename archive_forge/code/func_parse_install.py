import logging
from cmakelang import lex
from cmakelang.parse.common import KwargBreaker
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import PatternNode
from cmakelang.parse.util import (
def parse_install(ctx, tokens, breakstack):
    """
  The ``install()`` command has multiple different forms, implemented
  by different functions. The forms are indicated by the first token
  and are:

  * TARGETS
  * FILES
  * DIRECTORY
  * SCRIPT
  * CODE
  * EXPORT

  :see: https://cmake.org/cmake/help/v3.0/command/install.html
  """
    descriminator_token = get_first_semantic_token(tokens)
    if descriminator_token is None:
        logger.warning('Invalid install() command at %s', tokens[0].get_location())
        return StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={}, flags=[], breakstack=breakstack)
    descriminator = descriminator_token.spelling.upper()
    parsemap = {'TARGETS': parse_install_targets, 'FILES': parse_install_files, 'PROGRAMS': parse_install_files, 'DIRECTORY': parse_install_directory, 'SCRIPT': parse_install_script, 'CODE': parse_install_script, 'EXPORT': parse_install_export}
    if descriminator not in parsemap:
        logger.warning('Invalid install form "%s" at %s', descriminator, tokens[0].location())
        return StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={}, flags=[], breakstack=breakstack)
    return parsemap[descriminator](ctx, tokens, breakstack)