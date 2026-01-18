from __future__ import unicode_literals
import argparse
import collections
import io
import json
import logging
import os
import shutil
import sys
import cmakelang
from cmakelang import common
from cmakelang import configuration
from cmakelang import config_util
from cmakelang.format import formatter
from cmakelang import lex
from cmakelang import markup
from cmakelang import parse
from cmakelang.parse.argument_nodes import StandardParser2
from cmakelang.parse.common import NodeType, TreeNode
from cmakelang.parse.printer import dump_tree as dump_parse
from cmakelang.parse.funs import standard_funs
def setup_argparser(argparser):
    """
  Add argparse options to the parser.
  """
    argparser.register('action', 'extend', ExtendAction)
    argparser.add_argument('-v', '--version', action='version', version=cmakelang.__version__)
    argparser.add_argument('-l', '--log-level', default='info', choices=['error', 'warning', 'info', 'debug'])
    mutex = argparser.add_mutually_exclusive_group()
    mutex.add_argument('--dump-config', choices=['yaml', 'json', 'python'], default=None, const='python', nargs='?', help='If specified, print the default configuration to stdout and exit')
    mutex.add_argument('--dump', choices=['lex', 'parse', 'parsedb', 'layout', 'markup'], default=None)
    argparser.add_argument('--no-help', action='store_false', dest='with_help', help='When used with --dump-config, will omit helptext comments in the output')
    argparser.add_argument('--no-default', action='store_false', dest='with_defaults', help='When used with --dump-config, will omit any unmodified configuration value.')
    mutex = argparser.add_mutually_exclusive_group()
    mutex.add_argument('-i', '--in-place', action='store_true')
    mutex.add_argument('--check', action='store_true', help='Exit with status code 0 if formatting would not change file contents, or status code 1 if it would')
    mutex.add_argument('-o', '--outfile-path', default=None, help='Where to write the formatted file. Default is stdout.')
    argparser.add_argument('-c', '--config-files', nargs='+', action='extend', help='path to configuration file(s)')
    argparser.add_argument('infilepaths', nargs='*')
    configuration.Configuration().add_to_argparser(argparser)