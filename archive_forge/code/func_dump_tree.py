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
def dump_tree(nodes, outfile=None, indent=None):
    """
  Print a tree of node objects for debugging purposes
  """
    if indent is None:
        indent = ''
    if outfile is None:
        outfile = sys.stdout
    for idx, node in enumerate(nodes):
        outfile.write(indent)
        if idx + 1 == len(nodes):
            outfile.write('└─ ')
        else:
            outfile.write('├─ ')
        if sys.version_info[0] < 3:
            outfile.write(repr(node).decode('utf-8'))
        else:
            outfile.write(repr(node))
        outfile.write('\n')
        if not hasattr(node, 'children'):
            continue
        if idx + 1 == len(nodes):
            dump_tree(node.children, outfile, indent + '    ')
        else:
            dump_tree(node.children, outfile, indent + '│   ')