import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def coop_syncwarp(res):
    sm = cuda.shared.array(32, int32)
    i = cuda.grid(1)
    sm[i] = i
    cuda.syncwarp()
    if i < 16:
        sm[i] = sm[i] + sm[i + 16]
        cuda.syncwarp(65535)
    if i < 8:
        sm[i] = sm[i] + sm[i + 8]
        cuda.syncwarp(255)
    if i < 4:
        sm[i] = sm[i] + sm[i + 4]
        cuda.syncwarp(15)
    if i < 2:
        sm[i] = sm[i] + sm[i + 2]
        cuda.syncwarp(3)
    if i == 0:
        res[0] = sm[0] + sm[1]