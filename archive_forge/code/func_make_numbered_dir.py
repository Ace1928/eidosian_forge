from __future__ import annotations
import atexit
from contextlib import contextmanager
import fnmatch
import importlib.util
import io
import os
from os.path import abspath
from os.path import dirname
from os.path import exists
from os.path import isabs
from os.path import isdir
from os.path import isfile
from os.path import islink
from os.path import normpath
import posixpath
from stat import S_ISDIR
from stat import S_ISLNK
from stat import S_ISREG
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Literal
from typing import overload
from typing import TYPE_CHECKING
import uuid
import warnings
from . import error
@classmethod
def make_numbered_dir(cls, prefix='session-', rootdir=None, keep=3, lock_timeout=172800):
    """Return unique directory with a number greater than the current
        maximum one.  The number is assumed to start directly after prefix.
        if keep is true directories with a number less than (maxnum-keep)
        will be removed. If .lock files are used (lock_timeout non-zero),
        algorithm is multi-process safe.
        """
    if rootdir is None:
        rootdir = cls.get_temproot()
    nprefix = prefix.lower()

    def parse_num(path):
        """Parse the number out of a path (if it matches the prefix)"""
        nbasename = path.basename.lower()
        if nbasename.startswith(nprefix):
            try:
                return int(nbasename[len(nprefix):])
            except ValueError:
                pass

    def create_lockfile(path):
        """Exclusively create lockfile. Throws when failed"""
        mypid = os.getpid()
        lockfile = path.join('.lock')
        if hasattr(lockfile, 'mksymlinkto'):
            lockfile.mksymlinkto(str(mypid))
        else:
            fd = error.checked_call(os.open, str(lockfile), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 420)
            with os.fdopen(fd, 'w') as f:
                f.write(str(mypid))
        return lockfile

    def atexit_remove_lockfile(lockfile):
        """Ensure lockfile is removed at process exit"""
        mypid = os.getpid()

        def try_remove_lockfile():
            if os.getpid() != mypid:
                return
            try:
                lockfile.remove()
            except error.Error:
                pass
        atexit.register(try_remove_lockfile)
    lastmax = None
    while True:
        maxnum = -1
        for path in rootdir.listdir():
            num = parse_num(path)
            if num is not None:
                maxnum = max(maxnum, num)
        try:
            udir = rootdir.mkdir(prefix + str(maxnum + 1))
            if lock_timeout:
                lockfile = create_lockfile(udir)
                atexit_remove_lockfile(lockfile)
        except (error.EEXIST, error.ENOENT, error.EBUSY):
            if lastmax == maxnum:
                raise
            lastmax = maxnum
            continue
        break

    def get_mtime(path):
        """Read file modification time"""
        try:
            return path.lstat().mtime
        except error.Error:
            pass
    garbage_prefix = prefix + 'garbage-'

    def is_garbage(path):
        """Check if path denotes directory scheduled for removal"""
        bn = path.basename
        return bn.startswith(garbage_prefix)
    udir_time = get_mtime(udir)
    if keep and udir_time:
        for path in rootdir.listdir():
            num = parse_num(path)
            if num is not None and num <= maxnum - keep:
                try:
                    if lock_timeout:
                        create_lockfile(path)
                except (error.EEXIST, error.ENOENT, error.EBUSY):
                    path_time = get_mtime(path)
                    if not path_time:
                        continue
                    if abs(udir_time - path_time) < lock_timeout:
                        continue
                garbage_path = rootdir.join(garbage_prefix + str(uuid.uuid4()))
                try:
                    path.rename(garbage_path)
                    garbage_path.remove(rec=1)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    pass
            if is_garbage(path):
                try:
                    path.remove(rec=1)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    pass
    try:
        username = os.environ['USER']
    except KeyError:
        try:
            username = os.environ['USERNAME']
        except KeyError:
            username = 'current'
    src = str(udir)
    dest = src[:src.rfind('-')] + '-' + username
    try:
        os.unlink(dest)
    except OSError:
        pass
    try:
        os.symlink(src, dest)
    except (OSError, AttributeError, NotImplementedError):
        pass
    return udir