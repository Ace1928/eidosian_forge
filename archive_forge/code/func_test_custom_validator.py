from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_custom_validator(self):

    def make_parser():
        parser = ArgumentParser()
        parser.add_argument('var', choices=['bus', 'car'])
        parser.add_argument('value', choices=['orange', 'apple'])
        return parser
    expected_outputs = (('prog ', ['-h', '--help']), ('prog bu', ['']), ('prog bus ', ['-h', '--help']), ('prog bus appl', ['']), ('prog bus apple ', ['-h', '--help']))
    for cmd, output in expected_outputs:
        self.assertEqual(set(self.run_completer(make_parser(), cmd, validator=lambda x, y: False)), set(output))