from __future__ import annotations
from .. import http
def make_content_range(self, length):
    """Creates a :class:`~werkzeug.datastructures.ContentRange` object
        from the current range and given content length.
        """
    rng = self.range_for_length(length)
    if rng is not None:
        return ContentRange(self.units, rng[0], rng[1], length)
    return None