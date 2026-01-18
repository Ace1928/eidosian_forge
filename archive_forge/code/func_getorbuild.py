from time import time as gettime
def getorbuild(self, key, builder):
    try:
        entry = self._getentry(key)
    except KeyError:
        entry = self._build(key, builder)
        self._putentry(key, entry)
    return entry.value