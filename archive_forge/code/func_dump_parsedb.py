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
def dump_parsedb(parsedb, outfile=None, indent=None):
    """
  Dump the parse database to a file
  """
    if indent is None:
        indent = ''
    if outfile is None:
        outfile = sys.stdout
    items = list(sorted(parsedb.items()))
    for idx, (name, value) in enumerate(items):
        outfile.write(indent)
        if idx + 1 == len(items):
            outfile.write('└─ ')
            subindent = indent + '   '
        else:
            outfile.write('├─ ')
            subindent = indent + '|  '
        outfile.write(name)
        if isinstance(value, StandardParser2):
            outfile.write(': {}\n'.format(repr(value.cmdspec.pargs)))
            dump_parsedb(value.funtree, outfile, subindent)
        else:
            outfile.write(': {}\n'.format(type(value)))