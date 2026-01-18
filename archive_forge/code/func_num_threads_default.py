import platform
import sys
import os
import re
import shutil
import warnings
import traceback
import llvmlite.binding as ll
def num_threads_default():
    try:
        sched_getaffinity = os.sched_getaffinity
    except AttributeError:
        pass
    else:
        return max(1, len(sched_getaffinity(0)))
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        return max(1, cpu_count)
    return 1