from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_directory_completion(self):
    completer = DirectoriesCompleter()

    def c(prefix):
        return set(completer(prefix))
    with TempDir(prefix='test_dir', dir='.'):
        os.makedirs(os.path.join('abc', 'baz'))
        os.makedirs(os.path.join('abb', 'baz'))
        os.makedirs(os.path.join('abc', 'faz'))
        os.makedirs(os.path.join('def', 'baz'))
        with open('abc1', 'w') as fp1:
            with open('def1', 'w') as fp2:
                fp1.write('A test')
                fp2.write('Another test')
        self.assertEqual(c('a'), set(['abb/', 'abc/']))
        self.assertEqual(c('ab'), set(['abc/', 'abb/']))
        self.assertEqual(c('abc'), set(['abc/']))
        self.assertEqual(c('abc/'), set(['abc/baz/', 'abc/faz/']))
        self.assertEqual(c('d'), set(['def/']))
        self.assertEqual(c('def/'), set(['def/baz/']))
        self.assertEqual(c('e'), set([]))
        self.assertEqual(c('def/k'), set([]))
    return