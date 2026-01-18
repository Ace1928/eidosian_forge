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
def replace_with_tabs(content, config):
    """
  Apply the tab policy to `content`. If config.use_tabchars is true, then
  replace spaces at the beginning of every line with tabs according to
  config.tab_size and config.fractional_tab_policy.
  """
    if not config.format.use_tabchars:
        return content
    outlines = []
    for inline in content.split('\n'):
        num_spaces = count_indentation(inline)
        num_tabs, fractional = divmod(num_spaces, config.format.tab_size)
        outline = '\t' * num_tabs
        if fractional:
            if config.format.fractional_tab_policy == 'use-space':
                outline += ' ' * fractional
            elif config.format.fractional_tab_policy == 'round-up':
                outline += '\t'
            else:
                raise UserError("Invalid fractional_tab_policy='{}'".format(config.format.fractional_tab_policy))
        outline += inline[num_spaces:]
        outlines.append(outline)
    return '\n'.join(outlines)