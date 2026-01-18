import sys
import os.path
import csv
from docutils import io, nodes, statemachine, utils
from docutils.utils.error_reporting import SafeString
from docutils.utils import SystemMessagePropagation
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
def process_header_option(self):
    source = self.state_machine.get_source(self.lineno - 1)
    table_head = []
    max_header_cols = 0
    if 'header' in self.options:
        rows, max_header_cols = self.parse_csv_data_into_rows(self.options['header'].split('\n'), self.HeaderDialect(), source)
        table_head.extend(rows)
    return (table_head, max_header_cols)