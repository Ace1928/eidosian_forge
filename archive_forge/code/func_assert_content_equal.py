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
def assert_content_equal(self, a, b):
    with open(a, 'rb') as a_f:
        with open(b, 'rb') as b_f:
            assert a_f.read() == b_f.read()