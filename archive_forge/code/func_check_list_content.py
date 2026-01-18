import sys
import os.path
import csv
from docutils import io, nodes, statemachine, utils
from docutils.utils.error_reporting import SafeString
from docutils.utils import SystemMessagePropagation
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
def check_list_content(self, node):
    if len(node) != 1 or not isinstance(node[0], nodes.bullet_list):
        error = self.state_machine.reporter.error('Error parsing content block for the "%s" directive: exactly one bullet list expected.' % self.name, nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
        raise SystemMessagePropagation(error)
    list_node = node[0]
    for item_index in range(len(list_node)):
        item = list_node[item_index]
        if len(item) != 1 or not isinstance(item[0], nodes.bullet_list):
            error = self.state_machine.reporter.error('Error parsing content block for the "%s" directive: two-level bullet list expected, but row %s does not contain a second-level bullet list.' % (self.name, item_index + 1), nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
            raise SystemMessagePropagation(error)
        elif item_index:
            if len(item[0]) != num_cols:
                error = self.state_machine.reporter.error('Error parsing content block for the "%s" directive: uniform two-level bullet list expected, but row %s does not contain the same number of items as row 1 (%s vs %s).' % (self.name, item_index + 1, len(item[0]), num_cols), nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
                raise SystemMessagePropagation(error)
        else:
            num_cols = len(item[0])
    col_widths = self.get_column_widths(num_cols)
    return (num_cols, col_widths)