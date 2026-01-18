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
def makedev(self, tarinfo, targetpath):
    """Make a character or block device called targetpath.
        """
    if not hasattr(os, 'mknod') or not hasattr(os, 'makedev'):
        raise ExtractError('special devices not supported by system')
    mode = tarinfo.mode
    if mode is None:
        mode = 384
    if tarinfo.isblk():
        mode |= stat.S_IFBLK
    else:
        mode |= stat.S_IFCHR
    os.mknod(targetpath, mode, os.makedev(tarinfo.devmajor, tarinfo.devminor))