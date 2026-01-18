from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def mpc_ae(a, b, eps=eps):
    res = True
    res = res and a.real.ae(b.real, eps)
    res = res and a.imag.ae(b.imag, eps)
    return res