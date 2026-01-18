import os
import sys
import threading
from . import process
from . import reduction
def cpu_count(self):
    """Returns the number of CPUs in the system"""
    num = os.cpu_count()
    if num is None:
        raise NotImplementedError('cannot determine number of cpus')
    else:
        return num