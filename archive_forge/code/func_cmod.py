from __future__ import absolute_import
import math, sys
def cmod(a, b):
    r = a % b
    if a * b < 0 and r:
        r -= b
    return r