import os
import stat
import time
import datetime
import sys
import fnmatch
def check_sum():
    """
    Return a long which can be used to know if any .py files have changed.
    """
    val = 0
    for root, dirs, files in os.walk(os.getcwd()):
        for extension in EXTENSIONS:
            for f in fnmatch.filter(files, extension):
                stats = os.stat(os.path.join(root, f))
                val += stats[stat.ST_SIZE] + stats[stat.ST_MTIME]
    return val