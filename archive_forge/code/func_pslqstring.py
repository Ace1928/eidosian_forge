from .libmp.backend import xrange
from .libmp import int_types, sqrt_fixed
def pslqstring(r, constants):
    q = r[0]
    r = r[1:]
    s = []
    for i in range(len(r)):
        p = r[i]
        if p:
            z = fracgcd(-p, q)
            cs = constants[i][1]
            if cs == '1':
                cs = ''
            else:
                cs = '*' + cs
            if isinstance(z, int_types):
                if z > 0:
                    term = str(z) + cs
                else:
                    term = '(%s)' % z + cs
            else:
                term = '(%s/%s)' % z + cs
            s.append(term)
    s = ' + '.join(s)
    if '+' in s or '*' in s:
        s = '(' + s + ')'
    return s or '0'