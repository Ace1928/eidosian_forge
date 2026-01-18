from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def expect_node(self, root=False, sequence=False, mapping=False, simple_key=False):
    self.root_context = root
    self.sequence_context = sequence
    self.mapping_context = mapping
    self.simple_key_context = simple_key
    if isinstance(self.event, AliasEvent):
        self.expect_alias()
    elif isinstance(self.event, (ScalarEvent, CollectionStartEvent)):
        if self.process_anchor(u'&') and isinstance(self.event, ScalarEvent) and self.sequence_context:
            self.sequence_context = False
        self.process_tag()
        if isinstance(self.event, ScalarEvent):
            self.expect_scalar()
        elif isinstance(self.event, SequenceStartEvent):
            i2, n2 = (self.indention, self.no_newline)
            if self.event.comment:
                if self.event.flow_style is False and self.event.comment:
                    if self.write_post_comment(self.event):
                        self.indention = False
                        self.no_newline = True
                if self.write_pre_comment(self.event):
                    pass
                    self.indention = i2
                    self.no_newline = not self.indention
            if self.flow_level or self.canonical or self.event.flow_style or self.check_empty_sequence():
                self.expect_flow_sequence()
            else:
                self.expect_block_sequence()
        elif isinstance(self.event, MappingStartEvent):
            if self.event.flow_style is False and self.event.comment:
                self.write_post_comment(self.event)
            if self.event.comment and self.event.comment[1]:
                self.write_pre_comment(self.event)
            if self.flow_level or self.canonical or self.event.flow_style or self.check_empty_mapping():
                self.expect_flow_mapping(single=self.event.nr_items == 1)
            else:
                self.expect_block_mapping()
    else:
        raise EmitterError('expected NodeEvent, but got %s' % (self.event,))