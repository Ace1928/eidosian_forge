from mpmath import *
from mpmath.libmp import *
import random
def powr(x, n, r):
    return make_mpf(mpf_pow_int(x._mpf_, n, mp.prec, r))