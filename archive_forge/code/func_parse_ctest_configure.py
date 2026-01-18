from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_ctest_configure(ctx, tokens, breakstack):
    """
  ::

    ctest_configure([BUILD <build-dir>] [SOURCE <source-dir>] [APPEND]
                    [OPTIONS <options>] [RETURN_VALUE <result-var>] [QUIET]
                    [CAPTURE_CMAKE_ERROR <result-var>])

  :see: https://cmake.org/cmake/help/latest/command/ctest_configure.html
  """
    kwargs = {'BUILD': PositionalParser(1), 'SOURCE': PositionalParser(1), 'OPTIONS': PositionalParser(1), 'RETURN_VALUE': PositionalParser(1), 'CAPTURE_CMAKE_ERROR': PositionalParser(1)}
    return StandardArgTree.parse(ctx, tokens, '+', kwargs, ['APPEND', 'QUIET'], breakstack)