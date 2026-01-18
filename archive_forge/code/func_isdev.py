from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def isdev(self):
    """Return True if it is one of character device, block device or FIFO."""
    return self.type in (CHRTYPE, BLKTYPE, FIFOTYPE)