import math
import numpy as np
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import tensor_format
from tensorflow.python.debug.lib import common
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
def get_error_intro(tf_error):
    """Generate formatted intro for TensorFlow run-time error.

  Args:
    tf_error: (errors.OpError) TensorFlow run-time error object.

  Returns:
    (RichTextLines) Formatted intro message about the run-time OpError, with
      sample commands for debugging.
  """
    if hasattr(tf_error, 'op') and hasattr(tf_error.op, 'name'):
        op_name = tf_error.op.name
    else:
        op_name = None
    intro_lines = ['--------------------------------------', RL('!!! An error occurred during the run !!!', 'blink'), '']
    out = debugger_cli_common.rich_text_lines_from_rich_line_list(intro_lines)
    if op_name is not None:
        out.extend(debugger_cli_common.RichTextLines(['You may use the following commands to debug:']))
        out.extend(_recommend_command('ni -a -d -t %s' % op_name, 'Inspect information about the failing op.', create_link=True))
        out.extend(_recommend_command('li -r %s' % op_name, 'List inputs to the failing op, recursively.', create_link=True))
        out.extend(_recommend_command('lt', 'List all tensors dumped during the failing run() call.', create_link=True))
    else:
        out.extend(debugger_cli_common.RichTextLines(['WARNING: Cannot determine the name of the op that caused the error.']))
    more_lines = ['', 'Op name:    %s' % op_name, 'Error type: ' + str(type(tf_error)), '', 'Details:', str(tf_error), '', '--------------------------------------', '']
    out.extend(debugger_cli_common.RichTextLines(more_lines))
    return out