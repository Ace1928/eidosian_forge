from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def expect_block_mapping_value(self):
    self.write_indent()
    self.write_indicator(self.prefixed_colon, True, indention=True)
    self.states.append(self.expect_block_mapping_key)
    self.expect_node(mapping=True)