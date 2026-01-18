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
def map_merge(output_map, increment_map):
    """
  Merge `increment_map` into `output_map` recursively.
  """
    for key, increment_value in increment_map.items():
        if key not in output_map:
            output_map[key] = increment_value
            continue
        existing_value = output_map[key]
        if isinstance(existing_value, Mapping):
            if isinstance(increment_value, Mapping):
                map_merge(existing_value, increment_value)
            else:
                logger.warning('Cannot merge config %s of type %s into a dictionary', key, type(increment_value))
            continue
        output_map[key] = increment_value
    return output_map