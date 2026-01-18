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
def findCaller(self, stack_info=False, stacklevel=1):
    """Finds the frame of the calling method on the stack.

    This method skips any frames registered with the
    ABSLLogger and any methods from this file, and whatever
    method is currently being used to generate the prefix for the log
    line.  Then it returns the file name, line number, and method name
    of the calling method.  An optional fourth item may be returned,
    callers who only need things from the first three are advised to
    always slice or index the result rather than using direct unpacking
    assignment.

    Args:
      stack_info: bool, when True, include the stack trace as a fourth item
          returned.  On Python 3 there are always four items returned - the
          fourth will be None when this is False.  On Python 2 the stdlib
          base class API only returns three items.  We do the same when this
          new parameter is unspecified or False for compatibility.

    Returns:
      (filename, lineno, methodname[, sinfo]) of the calling method.
    """
    f_to_skip = ABSLLogger._frames_to_skip
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if _LOGGING_FILE_PREFIX not in code.co_filename and (code.co_filename, code.co_name, code.co_firstlineno) not in f_to_skip and ((code.co_filename, code.co_name) not in f_to_skip):
            if six.PY2 and (not stack_info):
                return (code.co_filename, frame.f_lineno, code.co_name)
            else:
                sinfo = None
                if stack_info:
                    out = io.StringIO()
                    out.write(u'Stack (most recent call last):\n')
                    traceback.print_stack(frame, file=out)
                    sinfo = out.getvalue().rstrip(u'\n')
                return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
        frame = frame.f_back