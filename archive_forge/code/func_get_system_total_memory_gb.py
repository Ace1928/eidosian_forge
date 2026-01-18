import os
import numpy as np
import threading
from time import time
from .. import config, logging
def get_system_total_memory_gb():
    """
    Function to get the total RAM of the running system in GB
    """
    import os
    import sys
    if 'linux' in sys.platform:
        with open('/proc/meminfo', 'r') as f_in:
            meminfo_lines = f_in.readlines()
            mem_total_line = [line for line in meminfo_lines if 'MemTotal' in line][0]
            mem_total = float(mem_total_line.split()[1])
            memory_gb = mem_total / 1024.0 ** 2
    elif 'darwin' in sys.platform:
        mem_str = os.popen('sysctl hw.memsize').read().strip().split(' ')[-1]
        memory_gb = float(mem_str) / 1024.0 ** 3
    else:
        err_msg = 'System platform: %s is not supported'
        raise Exception(err_msg)
    return memory_gb