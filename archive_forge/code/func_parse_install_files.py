from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_install_files(ctx, tokens, breakstack):
    """
  ::

    install_files(<dir> extension file file ...)
    install_files(<dir> regexp)
    install_files(<dir> FILES file file ...)

  :see: https://cmake.org/cmake/help/latest/command/install_files.html
  """
    second_token = get_nth_semantic_token(tokens, 1)
    third_token = get_nth_semantic_token(tokens, 2)
    if second_token is not None and second_token.spelling.upper() == 'FILES':
        return StandardArgTree.parse(ctx, tokens, '3+', {}, ['FILES'], breakstack)
    if third_token is None:
        return StandardArgTree.parse(ctx, tokens, 2, {}, [], breakstack)
    return StandardArgTree.parse(ctx, tokens, '3+', {}, [], breakstack)