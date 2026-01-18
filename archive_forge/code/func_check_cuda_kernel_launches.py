import os
import re
import sys
from typing import List
def check_cuda_kernel_launches():
    """Checks all pytorch code for CUDA kernel launches without cuda error checks

    Returns:
        The number of unsafe kernel launches in the codebase
    """
    torch_dir = os.path.dirname(os.path.realpath(__file__))
    torch_dir = os.path.dirname(torch_dir)
    torch_dir = os.path.dirname(torch_dir)
    kernels_without_checks = 0
    files_without_checks = []
    for root, dirnames, filenames in os.walk(torch_dir):
        if root == os.path.join(torch_dir, 'build') or root == os.path.join(torch_dir, 'torch/include'):
            dirnames[:] = []
            continue
        for x in filenames:
            filename = os.path.join(root, x)
            file_result = check_file(filename)
            if file_result > 0:
                kernels_without_checks += file_result
                files_without_checks.append(filename)
    if kernels_without_checks > 0:
        count_str = f"Found {kernels_without_checks} instances in {len(files_without_checks)} files where kernel launches didn't have checks."
        print(count_str, file=sys.stderr)
        print('Files without checks:', file=sys.stderr)
        for x in files_without_checks:
            print(f'\t{x}', file=sys.stderr)
        print(count_str, file=sys.stderr)
    return kernels_without_checks