import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from importlib import reload
from os.path import abspath, join
from unittest.mock import patch
import pytest
from tempfile import TemporaryDirectory
import IPython
from IPython import paths
from IPython.testing import decorators as dec
from IPython.testing.decorators import (
from IPython.testing.tools import make_tempfile
from IPython.utils import path
@skip_win32
def test_match_posix(self):
    for patterns, matches in self.common_cases() + [(['\\*'], ['*']), (['a\\*', 'a*'], ['a*'] + self.filenames_start_with_a), (['a\\[012]'], ['a[012]'])]:
        self.check_match(patterns, matches)