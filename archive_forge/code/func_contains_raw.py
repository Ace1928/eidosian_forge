from __future__ import annotations
from collections.abc import Collection
def contains_raw(self, etag):
    """When passed a quoted tag it will check if this tag is part of the
        set.  If the tag is weak it is checked against weak and strong tags,
        otherwise strong only."""
    from ..http import unquote_etag
    etag, weak = unquote_etag(etag)
    if weak:
        return self.contains_weak(etag)
    return self.contains(etag)