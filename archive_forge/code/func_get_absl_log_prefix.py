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
def get_absl_log_prefix(record):
    """Returns the absl log prefix for the log record.

  Args:
    record: logging.LogRecord, the record to get prefix for.
  """
    created_tuple = time.localtime(record.created)
    created_microsecond = int(record.created % 1.0 * 1000000.0)
    critical_prefix = ''
    level = record.levelno
    if _is_non_absl_fatal_record(record):
        level = logging.ERROR
        critical_prefix = _CRITICAL_PREFIX
    severity = converter.get_initial_for_level(level)
    return '%c%02d%02d %02d:%02d:%02d.%06d %5d %s:%d] %s' % (severity, created_tuple.tm_mon, created_tuple.tm_mday, created_tuple.tm_hour, created_tuple.tm_min, created_tuple.tm_sec, created_microsecond, _get_thread_id(), record.filename, record.lineno, critical_prefix)