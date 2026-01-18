from __future__ import annotations
from collections.abc import Collection
def is_weak(self, etag):
    """Check if an etag is weak."""
    return etag in self._weak