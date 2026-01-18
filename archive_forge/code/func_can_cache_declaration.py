from __future__ import annotations
import numbers
from .abstract import MaybeChannelBound, Object
from .exceptions import ContentDisallowed
from .serialization import prepare_accept_content
@property
def can_cache_declaration(self):
    if self.queue_arguments:
        expiring_queue = 'x-expires' in self.queue_arguments
    else:
        expiring_queue = False
    return not expiring_queue and (not self.auto_delete)