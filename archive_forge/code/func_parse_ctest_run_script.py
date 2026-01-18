from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_ctest_run_script(ctx, tokens, breakstack):
    """
  ::

    ctest_run_script([NEW_PROCESS] script_file_name script_file_name1
                     script_file_name2 ... [RETURN_VALUE var])

  :see: https://cmake.org/cmake/help/latest/command/ctest_run_script.html
  """
    kwargs = {'RETURN_VALUE': PositionalParser(1)}
    flags = ['NEW_PROCESS']
    return StandardArgTree.parse(ctx, tokens, '+', kwargs, flags, breakstack)