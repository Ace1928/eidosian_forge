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
def chown(self, tarinfo, targetpath, numeric_owner):
    """Set owner of targetpath according to tarinfo. If numeric_owner
           is True, use .gid/.uid instead of .gname/.uname. If numeric_owner
           is False, fall back to .gid/.uid when the search based on name
           fails.
        """
    if hasattr(os, 'geteuid') and os.geteuid() == 0:
        g = tarinfo.gid
        u = tarinfo.uid
        if not numeric_owner:
            try:
                if grp and tarinfo.gname:
                    g = grp.getgrnam(tarinfo.gname)[2]
            except KeyError:
                pass
            try:
                if pwd and tarinfo.uname:
                    u = pwd.getpwnam(tarinfo.uname)[2]
            except KeyError:
                pass
        if g is None:
            g = -1
        if u is None:
            u = -1
        try:
            if tarinfo.issym() and hasattr(os, 'lchown'):
                os.lchown(targetpath, u, g)
            else:
                os.chown(targetpath, u, g)
        except OSError as e:
            raise ExtractError('could not change owner') from e