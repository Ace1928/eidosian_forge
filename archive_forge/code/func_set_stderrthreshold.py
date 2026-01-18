from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import getpass
import io
import itertools
import logging
import os
import socket
import struct
import sys
import time
import timeit
import traceback
import types
import warnings
from absl import flags
from absl._collections_abc import abc
from absl.logging import converter
import six
def set_stderrthreshold(s):
    """Sets the stderr threshold to the value passed in.

  Args:
    s: str|int, valid strings values are case-insensitive 'debug',
        'info', 'warning', 'error', and 'fatal'; valid integer values are
        logging.DEBUG|INFO|WARNING|ERROR|FATAL.

  Raises:
      ValueError: Raised when s is an invalid value.
  """
    if s in converter.ABSL_LEVELS:
        FLAGS.stderrthreshold = converter.ABSL_LEVELS[s]
    elif isinstance(s, str) and s.upper() in converter.ABSL_NAMES:
        FLAGS.stderrthreshold = s
    else:
        raise ValueError('set_stderrthreshold only accepts integer absl logging level from -3 to 1, or case-insensitive string values \'debug\', \'info\', \'warning\', \'error\', and \'fatal\'. But found "{}" ({}).'.format(s, type(s)))