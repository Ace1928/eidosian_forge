import os
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
import numpy as np
import torch
def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a