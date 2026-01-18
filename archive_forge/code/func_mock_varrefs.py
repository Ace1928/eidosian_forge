import enum
import logging
import re
from cmakelang.common import InternalError
from cmakelang.format.formatter import get_comment_lines
from cmakelang.lex import TokenType, Token
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.body_nodes import BodyNode, FlowControlNode
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.common import NodeType, TreeNode
from cmakelang.parse.statement_node import StatementNode
from cmakelang.parse.util import get_min_npargs
from cmakelang.parse import variables
from cmakelang.parse.funs.set import SetFnNode
def mock_varrefs(tokenstr, repl=None):
    """Recursively replace variable references with a dummy string until all
     variable references are resolved.

  :see: https://cmake.org/cmake/help/v3.12/policy/CMP0053.html#policy:CMP0053
  """
    if repl is None:
        repl = 'foo'
    varref = re.compile('(?<!\\\\)(\\\\\\\\)*\\$\\{(?:(?:[A-Za-z0-9_./+-])|(?:\\\\[^A-Za-z0-9_./+-]))+\\}')
    while varref.search(tokenstr):
        tokenstr = varref.sub(make_varref_callback(repl), tokenstr)
    return tokenstr