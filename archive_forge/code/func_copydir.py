from __future__ import unicode_literals
import typing
import six
from . import errors
from .base import FS
from .copy import copy_dir, copy_file
from .error_tools import unwrap_errors
from .info import Info
from .path import abspath, join, normpath
def copydir(self, src_path, dst_path, create=False, preserve_time=False):
    src_fs, _src_path = self.delegate_path(src_path)
    dst_fs, _dst_path = self.delegate_path(dst_path)
    with unwrap_errors({_src_path: src_path, _dst_path: dst_path}):
        if not create and (not dst_fs.exists(_dst_path)):
            raise errors.ResourceNotFound(dst_path)
        if not src_fs.getinfo(_src_path).is_dir:
            raise errors.DirectoryExpected(src_path)
        copy_dir(src_fs, _src_path, dst_fs, _dst_path, preserve_time=preserve_time)