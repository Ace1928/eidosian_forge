from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_target_compile_features(ctx, tokens, breakstack):
    """
  ::

    target_compile_features(<target> <PRIVATE|PUBLIC|INTERFACE> <feature> [...])

  :see: https://cmake.org/cmake/help/latest/command/target_compile_features.html
  """
    kwargs = {'INTERFACE': PositionalParser('+'), 'PUBLIC': PositionalParser('+'), 'PRIVATE': PositionalParser('+')}
    return StandardArgTree.parse(ctx, tokens, 1, kwargs, [], breakstack)