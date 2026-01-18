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
def normalize_line_endings(instr):
    """
  Remove trailing whitespace and replace line endings with unix line endings.
  They will be replaced with config.endl during output
  """
    return re.sub('[ \t\x0c\x0b]*((\r?\n)|(\r\n?))', '\n', instr)