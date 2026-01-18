import sys
import json
from .symbols import *
from .symbols import Symbol
def emit_dict_diff(self, a, b, s, added, changed, removed):
    if s == 0.0:
        return [a, b]
    elif s == 1.0:
        return {}
    else:
        d = changed
        if added:
            d[insert] = added
        if removed:
            d[delete] = removed
        return d