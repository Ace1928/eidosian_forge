from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_action_activation_with_subparser(self):

    def make_parser():
        parser = ArgumentParser()
        parser.add_argument('name', nargs=2, choices=['a', 'b', 'c'])
        subparsers = parser.add_subparsers(title='subcommands', metavar='subcommand')
        subparser_build = subparsers.add_parser('build')
        subparser_build.add_argument('var', choices=['bus', 'car'])
        subparser_build.add_argument('--profile', nargs=1)
        return parser
    expected_outputs = (('prog ', ['a', 'b', 'c', '-h', '--help']), ('prog b', ['b ']), ('prog b ', ['a', 'b', 'c', '-h', '--help']), ('prog c b ', ['build', '-h', '--help']), ('prog c b bu', ['build ']), ('prog c b build ', ['bus', 'car', '--profile', '-h', '--help']), ('prog c b build ca', ['car ']), ('prog c b build car ', ['--profile', '-h', '--help']), ('prog build car ', ['-h', '--help']), ('prog a build car ', ['-h', '--help']))
    for cmd, output in expected_outputs:
        self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))