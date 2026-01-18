import unittest
import os
import sys
import sysconfig
from test.support import (
from distutils.command.config import dump_file, config
from distutils.tests import support
from distutils import log
def test_dump_file(self):
    this_file = os.path.splitext(__file__)[0] + '.py'
    f = open(this_file)
    try:
        numlines = len(f.readlines())
    finally:
        f.close()
    dump_file(this_file, 'I am the header')
    self.assertEqual(len(self._logs), numlines + 1)