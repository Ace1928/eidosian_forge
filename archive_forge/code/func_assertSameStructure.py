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
def assertSameStructure(self, a, b, aname='a', bname='b', msg=None):
    """Asserts that two values contain the same structural content.

    The two arguments should be data trees consisting of trees of dicts and
    lists. They will be deeply compared by walking into the contents of dicts
    and lists; other items will be compared using the == operator.
    If the two structures differ in content, the failure message will indicate
    the location within the structures where the first difference is found.
    This may be helpful when comparing large structures.

    Mixed Sequence and Set types are supported. Mixed Mapping types are
    supported, but the order of the keys will not be considered in the
    comparison.

    Args:
      a: The first structure to compare.
      b: The second structure to compare.
      aname: Variable name to use for the first structure in assertion messages.
      bname: Variable name to use for the second structure.
      msg: Additional text to include in the failure message.
    """
    problems = []
    _walk_structure_for_problems(a, b, aname, bname, problems, self.assertEqual, self.failureException)
    if self.maxDiff is not None:
        max_problems_to_show = self.maxDiff // 80
        if len(problems) > max_problems_to_show:
            problems = problems[0:max_problems_to_show - 1] + ['...']
    if problems:
        self.fail('; '.join(problems), msg)