from __future__ import annotations
from typing import Optional
def fuzzy_xor(args):
    """Return None if any element of args is not True or False, else
    True (if there are an odd number of True elements), else False."""
    t = f = 0
    for a in args:
        ai = fuzzy_bool(a)
        if ai:
            t += 1
        elif ai is False:
            f += 1
        else:
            return
    return t % 2 == 1