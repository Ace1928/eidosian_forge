import time
from mpmath import *
def last_digits(a):
    r = repr(a)
    s = str(a)
    m = 10
    r = r.replace(s[:-m], '')
    r = r.replace("mpf('", '').replace("')", '')
    num0 = 0
    for c in r:
        if c == '0':
            num0 += 1
        else:
            break
    b = float(int(r)) / 10 ** (len(r) - m)
    if b >= 10 ** m - 0.5:
        raise NotImplementedError
    n = int(round(b))
    sn = str(n)
    s = s[:-m] + '0' * num0 + sn
    return s[-20:]