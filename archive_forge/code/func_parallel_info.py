import torch
def parallel_info():
    """Returns detailed string with parallelization settings"""
    return torch._C._parallel_info()