import copy
import logging
from collections import deque, namedtuple
from botocore.compat import accepts_kwargs
from botocore.utils import EVENT_ALIASES
def emit_until_response(self, event_name, **kwargs):
    aliased_event_name = self._alias_event_name(event_name)
    return self._emitter.emit_until_response(aliased_event_name, **kwargs)