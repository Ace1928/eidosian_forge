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
def ranges_filter(x):
    r = np.zeros(x.shape, dtype=bool)
    for range_start, range_end in ranges:
        r = np.logical_or(r, np.logical_and(x >= range_start, x <= range_end))
    return r