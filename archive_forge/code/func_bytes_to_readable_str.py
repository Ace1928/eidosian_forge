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
def bytes_to_readable_str(num_bytes, include_b=False):
    """Generate a human-readable string representing number of bytes.

  The units B, kB, MB and GB are used.

  Args:
    num_bytes: (`int` or None) Number of bytes.
    include_b: (`bool`) Include the letter B at the end of the unit.

  Returns:
    (`str`) A string representing the number of bytes in a human-readable way,
      including a unit at the end.
  """
    if num_bytes is None:
        return str(num_bytes)
    if num_bytes < 1024:
        result = '%d' % num_bytes
    elif num_bytes < 1048576:
        result = '%.2fk' % (num_bytes / 1024.0)
    elif num_bytes < 1073741824:
        result = '%.2fM' % (num_bytes / 1048576.0)
    else:
        result = '%.2fG' % (num_bytes / 1073741824.0)
    if include_b:
        result += 'B'
    return result