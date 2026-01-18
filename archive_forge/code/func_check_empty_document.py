from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def check_empty_document(self):
    if not isinstance(self.event, DocumentStartEvent) or not self.events:
        return False
    event = self.events[0]
    return isinstance(event, ScalarEvent) and event.anchor is None and (event.tag is None) and event.implicit and (event.value == '')