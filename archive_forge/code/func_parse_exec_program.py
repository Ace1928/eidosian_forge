from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_exec_program(ctx, tokens, breakstack):
    """
  ::

    exec_program(Executable [directory in which to run]
                 [ARGS <arguments to executable>]
                 [OUTPUT_VARIABLE <var>]
                 [RETURN_VALUE <var>])

  :see: https://cmake.org/cmake/help/latest/command/exec_program.html
  """
    kwargs = {'ARGS': PositionalParser('+'), 'OUTPUT_VARIABLE': PositionalParser(1), 'RETURN_VALUE': PositionalParser(1)}
    return StandardArgTree.parse(ctx, tokens, 2, kwargs, [], breakstack)