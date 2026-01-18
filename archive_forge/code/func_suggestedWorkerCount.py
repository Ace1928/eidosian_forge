import multiprocessing
import os
import re
import sys
import time
from .processes import ForkedProcess
from .remoteproxy import ClosedError
@staticmethod
def suggestedWorkerCount():
    if 'linux' in sys.platform:
        try:
            cores = {}
            pid = None
            with open('/proc/cpuinfo') as fd:
                for line in fd:
                    m = re.match('physical id\\s+:\\s+(\\d+)', line)
                    if m is not None:
                        pid = m.groups()[0]
                    m = re.match('cpu cores\\s+:\\s+(\\d+)', line)
                    if m is not None:
                        cores[pid] = int(m.groups()[0])
            return sum(cores.values())
        except:
            return multiprocessing.cpu_count()
    else:
        return multiprocessing.cpu_count()