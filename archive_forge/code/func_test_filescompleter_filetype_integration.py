from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_filescompleter_filetype_integration(self):

    def make_parser():
        parser = ArgumentParser()
        parser.add_argument('--r', type=argparse.FileType('r'))
        parser.add_argument('--w', type=argparse.FileType('w'))
        return parser
    with TempDir(prefix='test_dir_fc2', dir='.'):
        os.makedirs(os.path.join('abcdefж', 'klm'))
        os.makedirs(os.path.join('abcaha', 'klm'))
        with open('abcxyz', 'w') as fh, open('abcdefж/klm/test', 'w') as fh2:
            fh.write('test')
            fh2.write('test')
        expected_outputs = (('prog subcommand --r ', ['abcxyz', 'abcdefж/', 'abcaha/']), ('prog subcommand --w abcdefж/klm/t', ['abcdefж/klm/test ']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))