from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_ctest_test(ctx, tokens, breakstack):
    """
  ::

    ctest_test([BUILD <build-dir>] [APPEND]
               [START <start-number>]
               [END <end-number>]
               [STRIDE <stride-number>]
               [EXCLUDE <exclude-regex>]
               [INCLUDE <include-regex>]
               [EXCLUDE_LABEL <label-exclude-regex>]
               [INCLUDE_LABEL <label-include-regex>]
               [EXCLUDE_FIXTURE <regex>]
               [EXCLUDE_FIXTURE_SETUP <regex>]
               [EXCLUDE_FIXTURE_CLEANUP <regex>]
               [PARALLEL_LEVEL <level>]
               [RESOURCE_SPEC_FILE <file>]
               [TEST_LOAD <threshold>]
               [SCHEDULE_RANDOM <ON|OFF>]
               [STOP_TIME <time-of-day>]
               [RETURN_VALUE <result-var>]
               [CAPTURE_CMAKE_ERROR <result-var>]
               [QUIET]
               )

  :see: https://cmake.org/cmake/help/latest/command/ctest_test.html
  """
    kwargs = {'BUILD': PositionalParser(1), 'START': PositionalParser(1), 'END': PositionalParser(1), 'STRIDE': PositionalParser(1), 'EXCLUDE': PositionalParser(1), 'INCLUDE': PositionalParser(1), 'EXCLUDE_LABEL': PositionalParser(1), 'INCLUDE_LABEL': PositionalParser(1), 'EXCLUDE_FIXTURE': PositionalParser(1), 'EXCLUDE_FIXTURE_SETUP': PositionalParser(1), 'EXCLUDE_FIXTURE_CLEANUP': PositionalParser(1), 'PARALLEL_LEVEL': PositionalParser(1), 'RESOURCE_SPEC_FILE': PositionalParser(1), 'TEST_LOAD': PositionalParser(1), 'SCHEDULE_RANDOM': PositionalParser(1, flags=['ON', 'OFF']), 'STOP_TIME': PositionalParser(1), 'RETURN_VALUE': PositionalParser(1), 'CAPTURE_CMAKE_ERROR': PositionalParser(1)}
    flags = ['APPEND', 'QUIET']
    return StandardArgTree.parse(ctx, tokens, '+', kwargs, flags, breakstack)