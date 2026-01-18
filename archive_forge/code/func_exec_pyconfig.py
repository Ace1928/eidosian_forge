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
def exec_pyconfig(configfile_path):
    _global = config_util.ExecGlobal(configfile_path)
    with io.open(configfile_path, 'r', encoding='utf-8') as infile:
        exec(infile.read(), _global)
    _global.pop('__builtins__', None)
    return _global