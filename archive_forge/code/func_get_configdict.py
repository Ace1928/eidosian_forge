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
def get_configdict(configfile_paths):
    include_queue = list(configfile_paths)
    config_dict = {}
    while include_queue:
        configfile_path = include_queue.pop(0)
        configfile_path = os.path.expanduser(configfile_path)
        increment_dict = get_one_config_dict(configfile_path)
        for include_path in increment_dict.pop('include', []):
            if not os.path.isabs(include_path):
                include_path = os.path.join(os.path.dirname(configfile_path), include_path)
            include_queue.append(include_path)
        map_merge(config_dict, increment_dict)
    return config_dict