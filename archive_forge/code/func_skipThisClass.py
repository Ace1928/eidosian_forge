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
def skipThisClass(reason):
    """Skip tests in the decorated TestCase, but not any of its subclasses.

  This decorator indicates that this class should skip all its tests, but not
  any of its subclasses. Useful for if you want to share testMethod or setUp
  implementations between a number of concrete testcase classes.

  Example usage, showing how you can share some common test methods between
  subclasses. In this example, only ``BaseTest`` will be marked as skipped, and
  not RealTest or SecondRealTest::

      @absltest.skipThisClass("Shared functionality")
      class BaseTest(absltest.TestCase):
        def test_simple_functionality(self):
          self.assertEqual(self.system_under_test.method(), 1)

      class RealTest(BaseTest):
        def setUp(self):
          super().setUp()
          self.system_under_test = MakeSystem(argument)

        def test_specific_behavior(self):
          ...

      class SecondRealTest(BaseTest):
        def setUp(self):
          super().setUp()
          self.system_under_test = MakeSystem(other_arguments)

        def test_other_behavior(self):
          ...

  Args:
    reason: The reason we have a skip in place. For instance: 'shared test
      methods' or 'shared assertion methods'.

  Returns:
    Decorator function that will cause a class to be skipped.
  """
    if isinstance(reason, type):
        raise TypeError('Got {!r}, expected reason as string'.format(reason))

    def _skip_class(test_case_class):
        if not issubclass(test_case_class, unittest.TestCase):
            raise TypeError('Decorating {!r}, expected TestCase subclass'.format(test_case_class))
        shadowed_setupclass = test_case_class.__dict__.get('setUpClass', None)

        @classmethod
        def replacement_setupclass(cls, *args, **kwargs):
            if cls is test_case_class:
                raise SkipTest(reason)
            if shadowed_setupclass:
                return shadowed_setupclass.__func__(cls, *args, **kwargs)
            else:
                return super(test_case_class, cls).setUpClass(*args, **kwargs)
        test_case_class.setUpClass = replacement_setupclass
        return test_case_class
    return _skip_class