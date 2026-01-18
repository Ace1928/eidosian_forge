from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def expect_scalar(self):
    self.increase_indent(flow=True)
    self.process_scalar()
    self.indent = self.indents.pop()
    self.state = self.states.pop()