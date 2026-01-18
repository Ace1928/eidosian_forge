import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
@cuda.jit
def random_3d(arr, rng_states):
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, stridez = cuda.gridsize(3)
    tid = startz * stridey * stridex + starty * stridex + startx
    for i in range(startz, arr.shape[0], stridez):
        for j in range(starty, arr.shape[1], stridey):
            for k in range(startx, arr.shape[2], stridex):
                arr[i, j, k] = xoroshiro128p_uniform_float32(rng_states, tid)