import sys
import json
from .symbols import *
from .symbols import Symbol
def emit_list_diff(self, a, b, s, inserted, changed, deleted):
    if s == 0.0:
        return [a, b]
    elif s == 1.0:
        return {}
    else:
        d = changed
        if inserted:
            d[insert] = inserted
        if deleted:
            d[delete] = deleted
        return d