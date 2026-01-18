import os
import io
import sys
import unittest
import warnings
import textwrap
from unittest import mock
from distutils.dist import Distribution, fix_help_options
from distutils.cmd import Command
from test.support import (
from test.support.os_helper import TESTFN
from distutils.tests import support
from distutils import log
def test_fix_help_options(self):
    help_tuples = [('a', 'b', 'c', 'd'), (1, 2, 3, 4)]
    fancy_options = fix_help_options(help_tuples)
    self.assertEqual(fancy_options[0], ('a', 'b', 'c'))
    self.assertEqual(fancy_options[1], (1, 2, 3))