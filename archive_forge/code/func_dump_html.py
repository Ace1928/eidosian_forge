from __future__ import unicode_literals
import io
import logging
import os
from cmakelang import lex
from cmakelang.parse.common import TreeNode
def dump_html(node, outfile):
    """
  Write to `outfile` an html annoted version of the listfile which has been
  parsed into the parse tree rooted at `node`
  """
    if isinstance(node, TreeNode):
        outfile.write('<span class="cmf-{}">'.format(node.node_type.name))
        for child in node.children:
            dump_html(child, outfile)
        outfile.write('</span>')
    elif isinstance(node, lex.Token):
        outfile.write('<span class="cmf-{}">'.format(node.type.name))
        outfile.write(node.spelling)
        outfile.write('</span>')