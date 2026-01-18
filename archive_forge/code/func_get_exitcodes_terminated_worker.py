import os
import sys
import time
import errno
import signal
import warnings
import subprocess
import traceback
def get_exitcodes_terminated_worker(processes):
    """Return a formatted string with the exitcodes of terminated workers.

    If necessary, wait (up to .25s) for the system to correctly set the
    exitcode of one terminated worker.
    """
    patience = 5
    exitcodes = [p.exitcode for p in list(processes.values()) if p.exitcode is not None]
    while not exitcodes and patience > 0:
        patience -= 1
        exitcodes = [p.exitcode for p in list(processes.values()) if p.exitcode is not None]
        time.sleep(0.05)
    return _format_exitcodes(exitcodes)