from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
def rules_2prereq(rules):
    """build prerequisites table from rules

       Description by example
       ----------------------

       given set of logic rules:

         a -> b, c
         b -> c

       we build prerequisites (from what points something can be deduced):

         b <- a
         c <- a, b

       rules:   {} of a -> [b, c, ...]
       return:  {} of c <- [a, b, ...]

       Note however, that this prerequisites may be *not* enough to prove a
       fact. An example is 'a -> b' rule, where prereq(a) is b, and prereq(b)
       is a. That's because a=T -> b=T, and b=F -> a=F, but a=F -> b=?
    """
    prereq = defaultdict(set)
    for (a, _), impl in rules.items():
        if isinstance(a, Not):
            a = a.args[0]
        for i, _ in impl:
            if isinstance(i, Not):
                i = i.args[0]
            prereq[i].add(a)
    return prereq