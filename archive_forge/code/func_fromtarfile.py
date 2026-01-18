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
@classmethod
def fromtarfile(cls, tarfile):
    """Return the next TarInfo object from TarFile object
           tarfile.
        """
    buf = tarfile.fileobj.read(BLOCKSIZE)
    obj = cls.frombuf(buf, tarfile.encoding, tarfile.errors)
    obj.offset = tarfile.fileobj.tell() - BLOCKSIZE
    return obj._proc_member(tarfile)