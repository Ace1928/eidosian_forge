from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def expect_first_flow_mapping_key(self):
    if isinstance(self.event, MappingEndEvent):
        self.indent = self.indents.pop()
        popped = self.flow_context.pop()
        assert popped == '{'
        self.write_indicator(u'}', False)
        if self.event.comment and self.event.comment[0]:
            self.write_post_comment(self.event)
        elif self.flow_level == 0:
            self.write_line_break()
        self.state = self.states.pop()
    else:
        if self.canonical or self.column > self.best_width:
            self.write_indent()
        if not self.canonical and self.check_simple_key():
            self.states.append(self.expect_flow_mapping_simple_value)
            self.expect_node(mapping=True, simple_key=True)
        else:
            self.write_indicator(u'?', True)
            self.states.append(self.expect_flow_mapping_value)
            self.expect_node(mapping=True)