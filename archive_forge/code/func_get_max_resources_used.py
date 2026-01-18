import os
import numpy as np
import threading
from time import time
from .. import config, logging
def get_max_resources_used(pid, mem_mb, num_threads, pyfunc=False):
    """
    Function to get the RAM and threads utilized by a given process

    Parameters
    ----------
    pid : integer
        the process ID of process to profile
    mem_mb : float
        the high memory watermark so far during process execution (in MB)
    num_threads: int
        the high thread watermark so far during process execution

    Returns
    -------
    mem_mb : float
        the new high memory watermark of process (MB)
    num_threads : float
        the new high thread watermark of process
    """
    if not resource_monitor:
        raise RuntimeError('Attempted to measure resources with option "monitoring.enabled" set off.')
    try:
        mem_mb = max(mem_mb, _get_ram_mb(pid, pyfunc=pyfunc))
        num_threads = max(num_threads, _get_num_threads(pid))
    except Exception as exc:
        proflogger.info('Could not get resources used by process.\n%s', exc)
    return (mem_mb, num_threads)