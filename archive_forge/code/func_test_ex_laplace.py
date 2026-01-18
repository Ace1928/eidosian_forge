import unittest
from numba.cuda.testing import (CUDATestCase, skip_if_cudadevrt_missing,
from numba.tests.support import captured_stdout
def test_ex_laplace(self):
    plot = False
    import numpy as np
    from numba import cuda
    size = 1001
    data = np.zeros(size)
    data[500] = 10000
    buf_0 = cuda.to_device(data)
    buf_1 = cuda.device_array_like(buf_0)
    niter = 10000
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(16 * 0.66, 9 * 0.66))
        plt.plot(np.arange(len(buf_0)), buf_0.copy_to_host(), lw=3, marker='*', color='black')
        plt.title('Initial State', fontsize=24)
        plt.xlabel('Position', fontsize=24)
        plt.ylabel('Temperature', fontsize=24)
        ax.set_xticks(ax.get_xticks(), fontsize=16)
        ax.set_yticks(ax.get_yticks(), fontsize=16)
        plt.xlim(0, len(data))
        plt.ylim(0, 10001)
        plt.savefig('laplace_initial.svg')

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
    solve_heat_equation.forall(len(data))(buf_0, buf_1, niter, 0.25)
    results = buf_1.copy_to_host()
    if plot:
        fig, ax = plt.subplots(figsize=(16 * 0.66, 9 * 0.66))
        plt.plot(np.arange(len(results)), results, lw=3, marker='*', color='black')
        plt.title(f'T = {niter}', fontsize=24)
        plt.xlabel('Position', fontsize=24)
        plt.ylabel('Temperature', fontsize=24)
        ax.set_xticks(ax.get_xticks(), fontsize=16)
        ax.set_yticks(ax.get_yticks(), fontsize=16)
        plt.ylim(0, max(results))
        plt.xlim(0, len(results))
        plt.savefig('laplace_final.svg')
    np.testing.assert_allclose(results.sum(), 10000)