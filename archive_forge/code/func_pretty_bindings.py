from __future__ import annotations
import numbers
from .abstract import MaybeChannelBound, Object
from .exceptions import ContentDisallowed
from .serialization import prepare_accept_content
def pretty_bindings(bindings):
    return '[{}]'.format(', '.join(map(str, bindings)))