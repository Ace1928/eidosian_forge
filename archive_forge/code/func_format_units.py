import copy
import operator
import re
import threading
def format_units(udict):
    """
    create a string representation of the units contained in a dimensionality
    """
    num = []
    den = []
    keys = [k for k, o in sorted(((k, k.format_order) for k in udict), key=operator.itemgetter(1))]
    for key in keys:
        d = udict[key]
        if config.use_unicode:
            u = key.u_symbol
        else:
            u = key.symbol
        if d > 0:
            if d != 1:
                u = u + ('**%s' % d).rstrip('0').rstrip('.')
            num.append(u)
        elif d < 0:
            d = -d
            if d != 1:
                u = u + ('**%s' % d).rstrip('0').rstrip('.')
            den.append(u)
    res = '*'.join(num)
    if len(den):
        if not res:
            res = '1'
        fmt = '(%s)' if len(den) > 1 else '%s'
        res = res + '/' + fmt % '*'.join(den)
    if not res:
        res = 'dimensionless'
    return res