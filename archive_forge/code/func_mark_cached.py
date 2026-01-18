import collections
import weakref
from tensorflow.python.util import object_identity
def mark_cached(self, key):
    self._set(key, True)