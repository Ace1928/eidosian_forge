import os
import subprocess
import io
import time
import threading
import Pyro4
def launch_ns():
    """Launches the pyro4-ns if it doesn't already exist.

    Returns the process.
    """
    return subprocess.Popen(['pyro4-ns'], shell=False)