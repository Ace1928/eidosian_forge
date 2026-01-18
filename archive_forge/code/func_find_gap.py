import functools
import warnings
import numpy as np
from ase.utils import IOContext
def find_gap(ev_k, ec_k, direct):
    """Helper function."""
    if direct:
        gap_k = ec_k - ev_k
        k = gap_k.argmin()
        return (gap_k[k], k, k)
    kv = ev_k.argmax()
    kc = ec_k.argmin()
    return (ec_k[kc] - ev_k[kv], kv, kc)