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
def assertContainsSubsequence(self, container, subsequence, msg=None):
    """Asserts that "container" contains "subsequence" as a subsequence.

    Asserts that "container" contains all the elements of "subsequence", in
    order, but possibly with other elements interspersed. For example, [1, 2, 3]
    is a subsequence of [0, 0, 1, 2, 0, 3, 0] but not of [0, 0, 1, 3, 0, 2, 0].

    Args:
      container: the list we're testing for subsequence inclusion.
      subsequence: the list we hope will be a subsequence of container.
      msg: Optional message to report on failure.
    """
    first_nonmatching = None
    reversed_container = list(reversed(container))
    subsequence = list(subsequence)
    for e in subsequence:
        if e not in reversed_container:
            first_nonmatching = e
            break
        while e != reversed_container.pop():
            pass
    if first_nonmatching is not None:
        self.fail('%s not a subsequence of %s. First non-matching element: %s' % (subsequence, container, first_nonmatching), msg)