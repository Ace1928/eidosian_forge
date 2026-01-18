import datetime
import io
import logging
import os
import os.path as osp
import re
import shutil
import stat
import tempfile
from fsspec import AbstractFileSystem
from fsspec.compression import compr
from fsspec.core import get_compression
from fsspec.utils import isfilelike, stringify_path
def info(self, path, **kwargs):
    if isinstance(path, os.DirEntry):
        out = path.stat(follow_symlinks=False)
        link = path.is_symlink()
        if path.is_dir(follow_symlinks=False):
            t = 'directory'
        elif path.is_file(follow_symlinks=False):
            t = 'file'
        else:
            t = 'other'
        path = self._strip_protocol(path.path)
    else:
        path = self._strip_protocol(path)
        out = os.stat(path, follow_symlinks=False)
        link = stat.S_ISLNK(out.st_mode)
        if link:
            out = os.stat(path, follow_symlinks=True)
        if stat.S_ISDIR(out.st_mode):
            t = 'directory'
        elif stat.S_ISREG(out.st_mode):
            t = 'file'
        else:
            t = 'other'
    result = {'name': path, 'size': out.st_size, 'type': t, 'created': out.st_ctime, 'islink': link}
    for field in ['mode', 'uid', 'gid', 'mtime', 'ino', 'nlink']:
        result[field] = getattr(out, f'st_{field}')
    if result['islink']:
        result['destination'] = os.readlink(path)
        try:
            out2 = os.stat(path, follow_symlinks=True)
            result['size'] = out2.st_size
        except OSError:
            result['size'] = 0
    return result