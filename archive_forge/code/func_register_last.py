import copy
import logging
from collections import deque, namedtuple
from botocore.compat import accepts_kwargs
from botocore.utils import EVENT_ALIASES
def register_last(self, event_name, handler, unique_id=None, unique_id_uses_count=False):
    aliased_event_name = self._alias_event_name(event_name)
    return self._emitter.register_last(aliased_event_name, handler, unique_id, unique_id_uses_count)