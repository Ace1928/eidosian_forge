from __future__ import annotations
import zlib
from kombu.utils.encoding import ensure_bytes
def encoders():
    """Return a list of available compression methods."""
    return list(_encoders)