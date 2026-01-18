from ._base import *
import operator as op
@property
def cache_file(self):
    return self._cachename + '.lazydb'