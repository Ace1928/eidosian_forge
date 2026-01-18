import os
import re
import sys
from typing import List
def check_code_for_cuda_kernel_launches(code, filename=None):
    """Checks code for CUDA kernel launches without cuda error checks.

    Args:
        filename - Filename of file containing the code. Used only for display
                   purposes, so you can put anything here.
        code     - The code to check

    Returns:
        The number of unsafe kernel launches in the code
    """
    if filename is None:
        filename = '##Python Function Call##'
    code = enumerate(code.split('\n'))
    code = [f'{lineno}: {linecode}' for lineno, linecode in code]
    code = '\n'.join(code)
    num_launches_without_checks = 0
    for m in kernel_launch_start.finditer(code):
        end_paren = find_matching_paren(code, m.end() - 1)
        if has_check.match(code, end_paren):
            num_launches_without_checks += 1
            context = code[m.start():end_paren + 1]
            print(f"Missing C10_CUDA_KERNEL_LAUNCH_CHECK in '{filename}'. Context:\n{context}", file=sys.stderr)
    return num_launches_without_checks