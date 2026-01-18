from __future__ import absolute_import, division, unicode_literals
from six import text_type
from bisect import bisect_left
from ._base import Trie as ABCTrie
def has_keys_with_prefix(self, prefix):
    if prefix in self._data:
        return True
    if prefix.startswith(self._cachestr):
        lo, hi = self._cachepoints
        i = bisect_left(self._keys, prefix, lo, hi)
    else:
        i = bisect_left(self._keys, prefix)
    if i == len(self._keys):
        return False
    return self._keys[i].startswith(prefix)