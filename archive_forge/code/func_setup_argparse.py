from __future__ import unicode_literals
import argparse
import io
import logging
import os
import sys
import cmakelang
from cmakelang import common
from cmakelang.format import __main__
from cmakelang import configuration
from cmakelang import lex
from cmakelang import parse
from cmakelang.lint import basic_checker
from cmakelang.lint import lint_util
def setup_argparse(argparser):
    argparser.add_argument('-v', '--version', action='version', version=cmakelang.__version__)
    argparser.add_argument('-l', '--log-level', default='info', choices=['error', 'warning', 'info', 'debug'])
    mutex = argparser.add_mutually_exclusive_group()
    mutex.add_argument('--dump-config', choices=['yaml', 'json', 'python'], default=None, const='python', nargs='?', help='If specified, print the default configuration to stdout and exit')
    mutex.add_argument('-o', '--outfile-path', default=None, help='Write errors to this file. Default is stdout.')
    argparser.add_argument('--no-help', action='store_false', dest='with_help', help='When used with --dump-config, will omit helptext comments in the output')
    argparser.add_argument('--no-default', action='store_false', dest='with_defaults', help='When used with --dump-config, will omit any unmodified configuration value.')
    argparser.add_argument('--suppress-decorations', action='store_true', help='Suppress the file title decoration and summary statistics')
    argparser.add_argument('-c', '--config-files', nargs='+', help='path to configuration file(s)')
    argparser.add_argument('infilepaths', nargs='*')
    configuration.Configuration().add_to_argparser(argparser)