from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def expect_flow_mapping(self, single=False):
    ind = self.indents.seq_flow_align(self.best_sequence_indent, self.column)
    map_init = u'{'
    if single and self.flow_level and (self.flow_context[-1] == '[') and (not self.canonical) and (not self.brace_single_entry_mapping_in_flow_sequence):
        map_init = u''
    self.write_indicator(u' ' * ind + map_init, True, whitespace=True)
    self.flow_context.append(map_init)
    self.increase_indent(flow=True, sequence=False)
    self.state = self.expect_first_flow_mapping_key