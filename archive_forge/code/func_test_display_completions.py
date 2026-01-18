from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_display_completions(self):
    parser = ArgumentParser()
    parser.add_argument('rover', choices=['sojourner', 'spirit', 'opportunity', 'curiosity'], help='help for rover ')
    parser.add_argument('antenna', choices=['low gain', 'high gain'], help='help for antenna')
    sub = parser.add_subparsers()
    p = sub.add_parser('list')
    p.add_argument('-o', '--oh', help='ttt')
    p.add_argument('-c', '--ch', help='ccc')
    sub2 = p.add_subparsers()
    sub2.add_parser('cat', help='list cat')
    sub2.add_parser('dog', help='list dog')
    completer = CompletionFinder(parser)
    completer.rl_complete('', 0)
    disp = completer.get_display_completions()
    self.assertEqual('help for rover ', disp.get('spirit', ''))
    self.assertEqual('help for rover ', disp.get('sojourner', ''))
    self.assertEqual('', disp.get('low gain', ''))
    completer.rl_complete('opportunity "low gain" list ', 0)
    disp = completer.get_display_completions()
    self.assertEqual('ttt', disp.get('-o --oh', ''))
    self.assertEqual('list cat', disp.get('cat', ''))
    completer.rl_complete('opportunity low\\ gain list --', 0)
    disp = completer.get_display_completions()
    self.assertEqual('ttt', disp.get('--oh', ''))
    self.assertEqual('ccc', disp.get('--ch', ''))