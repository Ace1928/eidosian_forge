from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def expect_block_mapping_key(self, first=False):
    if not first and isinstance(self.event, MappingEndEvent):
        if self.event.comment and self.event.comment[1]:
            self.write_pre_comment(self.event)
        self.indent = self.indents.pop()
        self.state = self.states.pop()
    else:
        if self.event.comment and self.event.comment[1]:
            self.write_pre_comment(self.event)
        self.write_indent()
        if self.check_simple_key():
            if not isinstance(self.event, (SequenceStartEvent, MappingStartEvent)):
                try:
                    if self.event.style == '?':
                        self.write_indicator(u'?', True, indention=True)
                except AttributeError:
                    pass
            self.states.append(self.expect_block_mapping_simple_value)
            self.expect_node(mapping=True, simple_key=True)
            if isinstance(self.event, AliasEvent):
                self.stream.write(u' ')
        else:
            self.write_indicator(u'?', True, indention=True)
            self.states.append(self.expect_block_mapping_value)
            self.expect_node(mapping=True)