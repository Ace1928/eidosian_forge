import sys
import os.path
import csv
from docutils import io, nodes, statemachine, utils
from docutils.utils.error_reporting import SafeString
from docutils.utils import SystemMessagePropagation
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
def get_column_widths(self, max_cols):
    if type(self.widths) == list:
        if len(self.widths) != max_cols:
            error = self.state_machine.reporter.error('"%s" widths do not match the number of columns in table (%s).' % (self.name, max_cols), nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
            raise SystemMessagePropagation(error)
        col_widths = self.widths
    elif max_cols:
        col_widths = [100 // max_cols] * max_cols
    else:
        error = self.state_machine.reporter.error('No table data detected in CSV file.', nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
        raise SystemMessagePropagation(error)
    return col_widths