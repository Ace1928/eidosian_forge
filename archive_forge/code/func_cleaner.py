import torch
import sys
import time
import os
import threading
import gc
def cleaner():
    nonlocal last_pos
    while True:
        time.sleep(0.1)
        pos = file.tell()
        if pos > last_pos:
            os.posix_fadvise(file.fileno(), last_pos, pos - last_pos, os.POSIX_FADV_DONTNEED)
        last_pos = pos