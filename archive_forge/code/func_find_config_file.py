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
def find_config_file(infile_path):
    """
  Search parent directories of an infile path and find a config file if
  one exists.
  """
    realpath = os.path.realpath(infile_path)
    if os.path.isdir(infile_path):
        head = infile_path
    else:
        head, _ = os.path.split(realpath)
    while head:
        for filename in ['.cmake-format', '.cmake-format.py', '.cmake-format.json', '.cmake-format.yaml', 'cmake-format.py', 'cmake-format.json', 'cmake-format.yaml']:
            configpath = os.path.join(head, filename)
            if os.path.exists(configpath):
                return configpath
        head2, _ = os.path.split(head)
        if head == head2:
            break
        head = head2
    return None