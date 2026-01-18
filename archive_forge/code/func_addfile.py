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
def addfile(self, tarinfo, fileobj=None):
    """Add the TarInfo object `tarinfo' to the archive. If `fileobj' is
           given, it should be a binary file, and tarinfo.size bytes are read
           from it and added to the archive. You can create TarInfo objects
           directly, or by using gettarinfo().
        """
    self._check('awx')
    tarinfo = copy.copy(tarinfo)
    buf = tarinfo.tobuf(self.format, self.encoding, self.errors)
    self.fileobj.write(buf)
    self.offset += len(buf)
    bufsize = self.copybufsize
    if fileobj is not None:
        copyfileobj(fileobj, self.fileobj, tarinfo.size, bufsize=bufsize)
        blocks, remainder = divmod(tarinfo.size, BLOCKSIZE)
        if remainder > 0:
            self.fileobj.write(NUL * (BLOCKSIZE - remainder))
            blocks += 1
        self.offset += blocks * BLOCKSIZE
    self.members.append(tarinfo)