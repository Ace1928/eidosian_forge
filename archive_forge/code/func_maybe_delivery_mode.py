from __future__ import annotations
import numbers
from .abstract import MaybeChannelBound, Object
from .exceptions import ContentDisallowed
from .serialization import prepare_accept_content
def maybe_delivery_mode(v, modes=None, default=PERSISTENT_DELIVERY_MODE):
    """Get delivery mode by name (or none if undefined)."""
    modes = DELIVERY_MODES if not modes else modes
    if v:
        return v if isinstance(v, numbers.Integral) else modes[v]
    return default