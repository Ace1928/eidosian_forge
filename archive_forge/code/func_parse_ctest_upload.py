from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_ctest_upload(ctx, tokens, breakstack):
    """
  ::

    ctest_upload(FILES <file>... [QUIET] [CAPTURE_CMAKE_ERROR <result-var>])

  :see: https://cmake.org/cmake/help/latest/command/ctest_upload.html
  """
    kwargs = {'FILES': PositionalParser('+'), 'CAPTURE_CMAKE_ERROR': PositionalParser(1)}
    flags = ['QUIET']
    return StandardArgTree.parse(ctx, tokens, '+', kwargs, flags, breakstack)