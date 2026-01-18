import os
import sys
def convert_ppid(ppid):
    ret = int(ppid)
    if ret != 0:
        if ret == os.getpid():
            raise AssertionError('ppid passed is the same as the current process pid (%s)!' % (ret,))
    return ret