import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout
@cuda.jit
def fast_matmul(A, B, C):
    """
            Perform matrix multiplication of C = A * B using CUDA shared memory.

            Reference: https://stackoverflow.com/a/64198479/13697228 by @RobertCrovella
            """
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x
    tmp = float32(0.0)
    for i in range(bpg):
        sA[ty, tx] = 0
        sB[ty, tx] = 0
        if y < A.shape[0] and tx + i * TPB < A.shape[1]:
            sA[ty, tx] = A[y, tx + i * TPB]
        if x < B.shape[1] and ty + i * TPB < B.shape[0]:
            sB[ty, tx] = B[ty + i * TPB, x]
        cuda.syncthreads()
        for j in range(TPB):
            tmp += sA[ty, j] * sB[j, tx]
        cuda.syncthreads()
    if y < C.shape[0] and x < C.shape[1]:
        C[y, x] = tmp