import contextlib
import decimal
import gc
import numpy as np
import os
import random
import re
import shutil
import signal
import socket
import string
import subprocess
import sys
import time
import pytest
import pyarrow as pa
import pyarrow.fs
def memory_leak_check(f, metric='rss', threshold=1 << 17, iterations=10, check_interval=1):
    """
    Execute the function and try to detect a clear memory leak either internal
    to Arrow or caused by a reference counting problem in the Python binding
    implementation. Raises exception if a leak detected

    Parameters
    ----------
    f : callable
        Function to invoke on each iteration
    metric : {'rss', 'vms', 'shared'}, default 'rss'
        Attribute of psutil.Process.memory_info to use for determining current
        memory use
    threshold : int, default 128K
        Threshold in number of bytes to consider a leak
    iterations : int, default 10
        Total number of invocations of f
    check_interval : int, default 1
        Number of invocations of f in between each memory use check
    """
    import psutil
    proc = psutil.Process()

    def _get_use():
        gc.collect()
        return getattr(proc.memory_info(), metric)
    baseline_use = _get_use()

    def _leak_check():
        current_use = _get_use()
        if current_use - baseline_use > threshold:
            raise Exception('Memory leak detected. Departure from baseline {} after {} iterations'.format(current_use - baseline_use, i))
    for i in range(iterations):
        f()
        if i % check_interval == 0:
            _leak_check()