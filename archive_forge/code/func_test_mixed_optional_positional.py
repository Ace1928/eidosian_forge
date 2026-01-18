from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_mixed_optional_positional(self):

    def make_parser():
        parser = ArgumentParser(add_help=False)
        parser.add_argument('name', choices=['name1', 'name2'])
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--get', action='store_true')
        group.add_argument('--set', action='store_true')
        return parser
    expected_outputs = (('prog ', ['--get', '--set', 'name1', 'name2']), ('prog --', ['--get', '--set']), ('prog --get ', ['--get', 'name1', 'name2']), ('prog --get name1 ', ['--get ']))
    for cmd, output in expected_outputs:
        self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))