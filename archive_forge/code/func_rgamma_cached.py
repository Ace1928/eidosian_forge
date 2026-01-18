from functools import lru_cache
import os
from time import time
import numpy as np
from scipy.special._mptestutils import mpf2float
@lru_cache(maxsize=100000)
def rgamma_cached(x, dps):
    with mp.workdps(dps):
        return mp.rgamma(x)