from copy import deepcopy
def frozendict_or(self, other, *args, **kwargs):
    res = {}
    res.update(self)
    res.update(other)
    return self.__class__(res)