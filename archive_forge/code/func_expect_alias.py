from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def expect_alias(self):
    if self.event.anchor is None:
        raise EmitterError('anchor is not specified for alias')
    self.process_anchor(u'*')
    self.state = self.states.pop()