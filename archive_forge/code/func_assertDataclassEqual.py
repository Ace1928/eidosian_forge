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
def assertDataclassEqual(self, first, second, msg=None):
    """Asserts two dataclasses are equal with more informative errors.

    Arguments must both be dataclasses. This compares equality of  individual
    fields and takes care to not compare fields that are marked as
    non-comparable. It gives per field differences, which are easier to parse
    than the comparison of the string representations from assertEqual.

    In cases where the dataclass has a custom __eq__, and it is defined in a
    way that is inconsistent with equality of comparable fields, we raise an
    exception without further trying to figure out how they are different.

    Args:
      first: A dataclass, the first value.
      second: A dataclass, the second value.
      msg: An optional str, the associated message.

    Raises:
      AssertionError: if the dataclasses are not equal.
    """
    if not dataclasses.is_dataclass(first) or isinstance(first, type):
        raise self.failureException('First argument is not a dataclass instance.')
    if not dataclasses.is_dataclass(second) or isinstance(second, type):
        raise self.failureException('Second argument is not a dataclass instance.')
    if first == second:
        return
    if type(first) is not type(second):
        self.fail('Found different dataclass types: %s != %s' % (type(first), type(second)), msg)
    different = [(f.name, getattr(first, f.name), getattr(second, f.name)) for f in dataclasses.fields(first) if f.compare and getattr(first, f.name) != getattr(second, f.name)]
    safe_repr = unittest.util.safe_repr
    message = ['%s != %s' % (safe_repr(first), safe_repr(second))]
    if different:
        message.append('Fields that differ:')
        message.extend(('%s: %s != %s' % (k, safe_repr(first_v), safe_repr(second_v)) for k, first_v, second_v in different))
    else:
        message.append('Cannot detect difference by examining the fields of the dataclass.')
    raise self.fail('\n'.join(message), msg)