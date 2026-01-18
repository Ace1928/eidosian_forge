import sys
import requests
def itermerged(self):
    """Iterate over all headers, merging duplicate ones together."""
    for key in self:
        val = self._container[key.lower()]
        yield (val[0], ', '.join(val[1:]))