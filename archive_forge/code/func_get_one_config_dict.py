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
def get_one_config_dict(configfile_path):
    """
  Return a dictionary of configuration options read from the given file path.
  If the filepath has a known extension then we parse it according to that
  extension. Otherwise we try to parse is using each parser one by one.
  """
    if not os.path.exists(configfile_path):
        raise common.UserError('Desired config file does not exist: {}'.format(configfile_path))
    if configfile_path.endswith('.json'):
        if os.stat(configfile_path).st_size == 0:
            return {}
        with io.open(configfile_path, 'r', encoding='utf-8') as config_file:
            try:
                return json.load(config_file)
            except ValueError as ex:
                message = 'Failed to parse json config file {}: {}'.format(configfile_path, ex)
                raise common.UserError(message)
    if configfile_path.endswith('.yaml'):
        with io.open(configfile_path, 'r', encoding='utf-8') as config_file:
            try:
                return load_yaml(config_file)
            except ValueError as ex:
                message = 'Failed to parse yaml config file {}: {}'.format(configfile_path, ex)
                raise common.UserError(message)
    if configfile_path.endswith('.py'):
        try:
            return exec_pyconfig(configfile_path)
        except Exception as ex:
            message = 'Failed to parse python config file {}: {}'.format(configfile_path, ex)
            raise common.UserError(message)
    return try_get_configdict(configfile_path)