from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
def deduce_alpha_implications(implications):
    """deduce all implications

       Description by example
       ----------------------

       given set of logic rules:

         a -> b
         b -> c

       we deduce all possible rules:

         a -> b, c
         b -> c


       implications: [] of (a,b)
       return:       {} of a -> set([b, c, ...])
    """
    implications = implications + [(Not(j), Not(i)) for i, j in implications]
    res = defaultdict(set)
    full_implications = transitive_closure(implications)
    for a, b in full_implications:
        if a == b:
            continue
        res[a].add(b)
    for a, impl in res.items():
        impl.discard(a)
        na = Not(a)
        if na in impl:
            raise ValueError('implications are inconsistent: %s -> %s %s' % (a, na, impl))
    return res