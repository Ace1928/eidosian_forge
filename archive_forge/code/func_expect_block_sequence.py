from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def expect_block_sequence(self):
    if self.mapping_context:
        indentless = not self.indention
    else:
        indentless = False
        if not self.compact_seq_seq and self.column != 0:
            self.write_line_break()
    self.increase_indent(flow=False, sequence=True, indentless=indentless)
    self.state = self.expect_first_block_sequence_item