from __future__ import absolute_import, division, unicode_literals
def longest_prefix(self, prefix):
    if prefix in self:
        return prefix
    for i in range(1, len(prefix) + 1):
        if prefix[:-i] in self:
            return prefix[:-i]
    raise KeyError(prefix)