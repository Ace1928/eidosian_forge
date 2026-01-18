import inspect
import weakref
def has(self, entity, subkey):
    key = self._get_key(entity)
    parent = self._cache.get(key, None)
    if parent is None:
        return False
    return subkey in parent