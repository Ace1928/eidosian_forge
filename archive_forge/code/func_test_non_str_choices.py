from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_non_str_choices(self):

    def make_parser():
        parser = ArgumentParser()
        parser.add_argument('x', type=int, choices=[4, 8, 15, 16, 23, 42])
        return parser
    expected_outputs = (('prog ', ['4', '8', '15', '16', '23', '42', '-h', '--help']), ('prog 1', ['15', '16']), ('prog 2', ['23 ']), ('prog 4', ['4', '42']), ('prog 4 ', ['-h', '--help']))
    for cmd, output in expected_outputs:
        self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))