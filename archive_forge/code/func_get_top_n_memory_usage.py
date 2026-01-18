import logging
import os
import platform
import sys
import time
import ray  # noqa F401
import psutil  # noqa E402
def get_top_n_memory_usage(n: int=10):
    """Get the top n memory usage of the process

    Params:
        n: Number of top n process memory usage to return.
    Returns:
        (str) The formatted string of top n process memory usage.
    """
    pids = psutil.pids()
    proc_stats = []
    for pid in pids:
        try:
            proc = psutil.Process(pid)
            proc_stats.append((get_rss(proc.memory_info()), pid, proc.cmdline()))
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            continue
    proc_str = 'PID\tMEM\tCOMMAND'
    for rss, pid, cmdline in sorted(proc_stats, reverse=True)[:n]:
        proc_str += '\n{}\t{}GiB\t{}'.format(pid, round(rss / 1024 ** 3, 2), ' '.join(cmdline)[:100].strip())
    return proc_str