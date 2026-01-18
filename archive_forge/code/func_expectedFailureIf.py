from collections import abc
import contextlib
import dataclasses
import difflib
import enum
import errno
import faulthandler
import getpass
import inspect
import io
import itertools
import json
import os
import random
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import typing
from typing import Any, AnyStr, BinaryIO, Callable, ContextManager, IO, Iterator, List, Mapping, MutableMapping, MutableSequence, NoReturn, Optional, Sequence, Text, TextIO, Tuple, Type, Union
import unittest
from unittest import mock  # pylint: disable=unused-import Allow absltest.mock.
from urllib import parse
from absl import app  # pylint: disable=g-import-not-at-top
from absl import flags
from absl import logging
from absl.testing import _pretty_print_reporter
from absl.testing import xml_reporter
def expectedFailureIf(condition, reason):
    """Expects the test to fail if the run condition is True.

  Example usage::

      @expectedFailureIf(sys.version.major == 2, "Not yet working in py2")
      def test_foo(self):
        ...

  Args:
    condition: bool, whether to expect failure or not.
    reason: Text, the reason to expect failure.
  Returns:
    Decorator function
  """
    del reason
    if condition:
        return unittest.expectedFailure
    else:
        return lambda f: f