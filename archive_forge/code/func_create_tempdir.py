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
def create_tempdir(self, name=None, cleanup=None):
    """Create a temporary directory specific to the test.

    NOTE: The directory and its contents will be recursively cleared before
    creation. This ensures that there is no pre-existing state.

    This creates a named directory on disk that is isolated to this test, and
    will be properly cleaned up by the test. This avoids several pitfalls of
    creating temporary directories for test purposes, as well as makes it easier
    to setup directories and verify their contents. For example::

        def test_foo(self):
          out_dir = self.create_tempdir()
          out_log = out_dir.create_file('output.log')
          expected_outputs = [
              os.path.join(out_dir, 'data-0.txt'),
              os.path.join(out_dir, 'data-1.txt'),
          ]
          code_under_test(out_dir)
          self.assertTrue(os.path.exists(expected_paths[0]))
          self.assertTrue(os.path.exists(expected_paths[1]))
          self.assertEqual('foo', out_log.read_text())

    See also: :meth:`create_tempfile` for creating temporary files.

    Args:
      name: Optional name of the directory. If not given, a unique
        name will be generated and used.
      cleanup: Optional cleanup policy on when/if to remove the directory (and
        all its contents) at the end of the test. If None, then uses
        :attr:`tempfile_cleanup`.

    Returns:
      A _TempDir representing the created directory; see _TempDir class docs
      for usage.
    """
    test_path = self._get_tempdir_path_test()
    if name:
        path = os.path.join(test_path, name)
        cleanup_path = os.path.join(test_path, _get_first_part(name))
    else:
        os.makedirs(test_path, exist_ok=True)
        path = tempfile.mkdtemp(dir=test_path)
        cleanup_path = path
    _rmtree_ignore_errors(cleanup_path)
    os.makedirs(path, exist_ok=True)
    self._maybe_add_temp_path_cleanup(cleanup_path, cleanup)
    return _TempDir(path)