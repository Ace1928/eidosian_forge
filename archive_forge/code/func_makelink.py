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
def makelink(self, tarinfo, targetpath):
    """Make a (symbolic) link called targetpath. If it cannot be created
          (platform limitation), we try to make a copy of the referenced file
          instead of a link.
        """
    try:
        if tarinfo.issym():
            if os.path.lexists(targetpath):
                os.unlink(targetpath)
            os.symlink(tarinfo.linkname, targetpath)
        elif os.path.exists(tarinfo._link_target):
            os.link(tarinfo._link_target, targetpath)
        else:
            self._extract_member(self._find_link_target(tarinfo), targetpath)
    except symlink_exception:
        try:
            self._extract_member(self._find_link_target(tarinfo), targetpath)
        except KeyError:
            raise ExtractError('unable to resolve link inside archive') from None