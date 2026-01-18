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
def test_command_packages_configfile(self):
    sys.argv.append('build')
    self.addCleanup(os.unlink, TESTFN)
    f = open(TESTFN, 'w')
    try:
        print('[global]', file=f)
        print('command_packages = foo.bar, splat', file=f)
    finally:
        f.close()
    d = self.create_distribution([TESTFN])
    self.assertEqual(d.get_command_packages(), ['distutils.command', 'foo.bar', 'splat'])
    sys.argv[1:] = ['--command-packages', 'spork', 'build']
    d = self.create_distribution([TESTFN])
    self.assertEqual(d.get_command_packages(), ['distutils.command', 'spork'])
    sys.argv[1:] = ['--command-packages', '', 'build']
    d = self.create_distribution([TESTFN])
    self.assertEqual(d.get_command_packages(), ['distutils.command'])