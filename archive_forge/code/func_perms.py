from greenlet import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def perms(l):
    if len(l) > 1:
        for e in l:
            x = [Yield([e] + p) for p in perms([x for x in l if x != e])]
            assert x
    else:
        Yield(l)