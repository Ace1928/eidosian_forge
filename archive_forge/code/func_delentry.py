from time import time as gettime
def delentry(self, key, raising=False):
    try:
        del self._dict[key]
    except KeyError:
        if raising:
            raise