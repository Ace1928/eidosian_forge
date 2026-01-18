import unittest
from numba.cuda.testing import (CUDATestCase, skip_if_cudadevrt_missing,
from numba.tests.support import captured_stdout
@cuda.jit
def solve_heat_equation(buf_0, buf_1, timesteps, k):
    i = cuda.grid(1)
    if i >= len(buf_0):
        return
    grid = cuda.cg.this_grid()
    for step in range(timesteps):
        if step % 2 == 0:
            data = buf_0
            next_data = buf_1
        else:
            data = buf_1
            next_data = buf_0
        curr_temp = data[i]
        if i == 0:
            next_temp = curr_temp + k * (data[i + 1] - 2 * curr_temp)
        elif i == len(data) - 1:
            next_temp = curr_temp + k * (data[i - 1] - 2 * curr_temp)
        else:
            next_temp = curr_temp + k * (data[i - 1] - 2 * curr_temp + data[i + 1])
        next_data[i] = next_temp
        grid.sync()