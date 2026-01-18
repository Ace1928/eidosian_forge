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
def gettarinfo(self, name=None, arcname=None, fileobj=None):
    """Create a TarInfo object from the result of os.stat or equivalent
           on an existing file. The file is either named by `name', or
           specified as a file object `fileobj' with a file descriptor. If
           given, `arcname' specifies an alternative name for the file in the
           archive, otherwise, the name is taken from the 'name' attribute of
           'fileobj', or the 'name' argument. The name should be a text
           string.
        """
    self._check('awx')
    if fileobj is not None:
        name = fileobj.name
    if arcname is None:
        arcname = name
    drv, arcname = os.path.splitdrive(arcname)
    arcname = arcname.replace(os.sep, '/')
    arcname = arcname.lstrip('/')
    tarinfo = self.tarinfo()
    tarinfo.tarfile = self
    if fileobj is None:
        if not self.dereference:
            statres = os.lstat(name)
        else:
            statres = os.stat(name)
    else:
        statres = os.fstat(fileobj.fileno())
    linkname = ''
    stmd = statres.st_mode
    if stat.S_ISREG(stmd):
        inode = (statres.st_ino, statres.st_dev)
        if not self.dereference and statres.st_nlink > 1 and (inode in self.inodes) and (arcname != self.inodes[inode]):
            type = LNKTYPE
            linkname = self.inodes[inode]
        else:
            type = REGTYPE
            if inode[0]:
                self.inodes[inode] = arcname
    elif stat.S_ISDIR(stmd):
        type = DIRTYPE
    elif stat.S_ISFIFO(stmd):
        type = FIFOTYPE
    elif stat.S_ISLNK(stmd):
        type = SYMTYPE
        linkname = os.readlink(name)
    elif stat.S_ISCHR(stmd):
        type = CHRTYPE
    elif stat.S_ISBLK(stmd):
        type = BLKTYPE
    else:
        return None
    tarinfo.name = arcname
    tarinfo.mode = stmd
    tarinfo.uid = statres.st_uid
    tarinfo.gid = statres.st_gid
    if type == REGTYPE:
        tarinfo.size = statres.st_size
    else:
        tarinfo.size = 0
    tarinfo.mtime = statres.st_mtime
    tarinfo.type = type
    tarinfo.linkname = linkname
    if pwd:
        try:
            tarinfo.uname = pwd.getpwuid(tarinfo.uid)[0]
        except KeyError:
            pass
    if grp:
        try:
            tarinfo.gname = grp.getgrgid(tarinfo.gid)[0]
        except KeyError:
            pass
    if type in (CHRTYPE, BLKTYPE):
        if hasattr(os, 'major') and hasattr(os, 'minor'):
            tarinfo.devmajor = os.major(statres.st_rdev)
            tarinfo.devminor = os.minor(statres.st_rdev)
    return tarinfo