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
def parse_ranges_highlight(ranges_string):
    """Process ranges highlight string.

  Args:
    ranges_string: (str) A string representing a numerical range of a list of
      numerical ranges. See the help info of the -r flag of the print_tensor
      command for more details.

  Returns:
    An instance of tensor_format.HighlightOptions, if range_string is a valid
      representation of a range or a list of ranges.
  """
    ranges = None

    def ranges_filter(x):
        r = np.zeros(x.shape, dtype=bool)
        for range_start, range_end in ranges:
            r = np.logical_or(r, np.logical_and(x >= range_start, x <= range_end))
        return r
    if ranges_string:
        ranges = command_parser.parse_ranges(ranges_string)
        return tensor_format.HighlightOptions(ranges_filter, description=ranges_string)
    else:
        return None