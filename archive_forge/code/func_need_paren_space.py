from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import io
import logging
import re
import sys
from cmakelang import lex
from cmakelang import markup
from cmakelang.common import UserError
from cmakelang.lex import TokenType
from cmakelang.parse.argument_nodes import PositionalGroupNode
from cmakelang.parse.common import FlowType, NodeType, TreeNode
from cmakelang.parse.util import comment_is_tag
from cmakelang.parse import simple_nodes
def need_paren_space(spelling, config):
    """
  Return whether or not we need a space between the statement name and the
  starting parenthesis. This aggregates the logic of the two configuration
  options `separate_ctrl_name_with_space` and `separate_fn_name_with_space`.
  """
    upper = spelling.upper()
    if FlowType.get(upper) is not None:
        return config.format.separate_ctrl_name_with_space
    if upper.startswith('END') and FlowType.get(upper[3:]) is not None:
        return config.format.separate_ctrl_name_with_space
    if upper in ['ELSE', 'ELSEIF']:
        return config.format.separate_ctrl_name_with_space
    return config.format.separate_fn_name_with_space