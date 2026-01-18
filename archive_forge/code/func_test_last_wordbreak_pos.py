from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_last_wordbreak_pos(self):
    self.assertEqual(self.wordbreak('a'), None)
    self.assertEqual(self.wordbreak('a b:c'), 1)
    self.assertEqual(self.wordbreak('a b:c=d'), 3)
    self.assertEqual(self.wordbreak('a b:c=d '), None)
    self.assertEqual(self.wordbreak('a b:c=d e'), None)
    self.assertEqual(self.wordbreak('"b:c'), None)
    self.assertEqual(self.wordbreak('"b:c=d'), None)
    self.assertEqual(self.wordbreak('"b:c=d"'), None)
    self.assertEqual(self.wordbreak('"b:c=d" '), None)