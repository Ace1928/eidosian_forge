from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_append_space(self):

    def make_parser():
        parser = ArgumentParser(add_help=False)
        parser.add_argument('foo', choices=['bar'])
        return parser
    self.assertEqual(self.run_completer(make_parser(), 'prog '), ['bar '])
    self.assertEqual(self.run_completer(make_parser(), 'prog ', append_space=False), ['bar'])