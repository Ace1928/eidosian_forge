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
@classmethod
def register_frame_to_skip(cls, file_name, function_name, line_number=None):
    """Registers a function name to skip when walking the stack.

    The ABSLLogger sometimes skips method calls on the stack
    to make the log messages meaningful in their appropriate context.
    This method registers a function from a particular file as one
    which should be skipped.

    Args:
      file_name: str, the name of the file that contains the function.
      function_name: str, the name of the function to skip.
      line_number: int, if provided, only the function with this starting line
          number will be skipped. Otherwise, all functions with the same name
          in the file will be skipped.
    """
    if line_number is not None:
        cls._frames_to_skip.add((file_name, function_name, line_number))
    else:
        cls._frames_to_skip.add((file_name, function_name))