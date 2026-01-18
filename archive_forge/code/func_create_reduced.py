import operator
import sys
from .libmp import int_types, mpf_hash, bitcount, from_man_exp, HASH_MODULUS
def create_reduced(p, q, _cache={}):
    key = (p, q)
    if key in _cache:
        return _cache[key]
    x, y = (p, q)
    while y:
        x, y = (y, x % y)
    if x != 1:
        p //= x
        q //= x
    v = new(mpq)
    v._mpq_ = (p, q)
    if q <= 4 and abs(key[0]) < 100:
        _cache[key] = v
    return v