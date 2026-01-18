from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_exclusive_class(self):
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--foo', dest='types', action='append_const', const=str)
    parser.add_argument('--bar', dest='types', action='append', choices=['bar1', 'bar2'])
    parser.add_argument('--baz', choices=['baz1', 'baz2'])
    parser.add_argument('--no-bar', action='store_true')
    completer = ExclusiveCompletionFinder(parser, always_complete_options=True)
    expected_outputs = (('prog ', ['--foo', '--bar', '--baz', '--no-bar']), ('prog --baz ', ['baz1', 'baz2']), ('prog --baz baz1 ', ['--foo', '--bar', '--no-bar']), ('prog --foo --no-bar ', ['--foo', '--bar', '--baz']), ('prog --foo --bar bar1 ', ['--foo', '--bar', '--baz', '--no-bar']))
    for cmd, output in expected_outputs:
        self.assertEqual(set(self.run_completer(parser, cmd, completer=completer)), set(output))