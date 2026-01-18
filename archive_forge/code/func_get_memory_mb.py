import os
import argparse
def get_memory_mb(mem_str):
    """Get the memory in MB from memory string.

    mem_str: str
        String representation of memory requirement.

    Returns
    -------
    mem_mb: int
        Memory requirement in MB.
    """
    mem_str = mem_str.lower()
    if mem_str.endswith('g'):
        return int(float(mem_str[:-1]) * 1024)
    elif mem_str.endswith('m'):
        return int(float(mem_str[:-1]))
    else:
        msg = 'Invalid memory specification %s, need to be a number follows g or m' % mem_str
        raise RuntimeError(msg)