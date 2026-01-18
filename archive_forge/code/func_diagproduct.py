import numpy as np
from numba import cuda, float32, void
from numba.cuda.testing import unittest, CUDATestCase
@cuda.jit(void(float32[:, :], float32[:, :], float32[:]))
def diagproduct(c, a, b):
    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y
    height = c.shape[0]
    width = c.shape[1]
    for x in range(startX, width, gridX):
        for y in range(startY, height, gridY):
            c[y, x] = a[y, x] * b[x]