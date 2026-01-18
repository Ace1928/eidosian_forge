from copy import deepcopy
def frozendict_reversed(self, *args, **kwargs):
    return reversed(tuple(self))