from __future__ import absolute_import, division, unicode_literals
def longest_prefix_item(self, prefix):
    lprefix = self.longest_prefix(prefix)
    return (lprefix, self[lprefix])