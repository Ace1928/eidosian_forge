from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_exclusive(self):

    def make_parser():
        parser = ArgumentParser(add_help=False)
        parser.add_argument('--foo', action='store_true')
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--bar', action='store_true')
        group.add_argument('--no-bar', action='store_true')
        return parser
    expected_outputs = (('prog ', ['--foo', '--bar', '--no-bar']), ('prog --foo ', ['--foo', '--bar', '--no-bar']), ('prog --bar ', ['--foo', '--bar']), ('prog --foo --no-bar ', ['--foo', '--no-bar']))
    for cmd, output in expected_outputs:
        self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))