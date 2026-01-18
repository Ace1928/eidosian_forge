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
def next_sibling(self):
    if self._parent is None:
        return None
    next_idx = self._index_in_parent() + 1
    if next_idx >= len(self._parent.children):
        return None
    return self._parent.children[next_idx]