from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def check_empty_sequence(self):
    return isinstance(self.event, SequenceStartEvent) and bool(self.events) and isinstance(self.events[0], SequenceEndEvent)