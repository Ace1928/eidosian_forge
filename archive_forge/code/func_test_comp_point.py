from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_comp_point(self):
    self.assertEqual(self.sh.run_command('export POINT=1'), '')
    self.assertEqual(self.sh.run_command('prog point hi\t'), '13\r\n')
    self.assertEqual(self.sh.run_command('prog point hi \t'), '14\r\n')
    self.assertEqual(self.sh.run_command('prog point 你好嘚瑟\t'), '15\r\n')
    self.assertEqual(self.sh.run_command('prog point 你好嘚瑟 \t'), '16\r\n')