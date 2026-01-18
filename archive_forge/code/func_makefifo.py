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
def makefifo(self, tarinfo, targetpath):
    """Make a fifo called targetpath.
        """
    if hasattr(os, 'mkfifo'):
        os.mkfifo(targetpath)
    else:
        raise ExtractError('fifo not supported by system')