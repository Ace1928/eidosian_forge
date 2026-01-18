from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def expect_first_flow_sequence_item(self):
    if isinstance(self.event, SequenceEndEvent):
        self.indent = self.indents.pop()
        popped = self.flow_context.pop()
        assert popped == '['
        self.write_indicator(u']', False)
        if self.event.comment and self.event.comment[0]:
            self.write_post_comment(self.event)
        elif self.flow_level == 0:
            self.write_line_break()
        self.state = self.states.pop()
    else:
        if self.canonical or self.column > self.best_width:
            self.write_indent()
        self.states.append(self.expect_flow_sequence_item)
        self.expect_node(sequence=True)