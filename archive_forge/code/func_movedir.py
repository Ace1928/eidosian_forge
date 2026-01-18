from __future__ import absolute_import, unicode_literals
import typing
import contextlib
import io
import os
import six
import time
from collections import OrderedDict
from threading import RLock
from . import errors
from ._typing import overload
from .base import FS
from .copy import copy_modified_time
from .enums import ResourceType, Seek
from .info import Info
from .mode import Mode
from .path import iteratepath, normpath, split
def movedir(self, src_path, dst_path, create=False, preserve_time=False):
    src_dir, src_name = split(self.validatepath(src_path))
    dst_dir, dst_name = split(self.validatepath(dst_path))
    with self._lock:
        src_dir_entry = self._get_dir_entry(src_dir)
        if src_dir_entry is None or src_name not in src_dir_entry:
            raise errors.ResourceNotFound(src_path)
        src_entry = src_dir_entry.get_entry(src_name)
        if not src_entry.is_dir:
            raise errors.DirectoryExpected(src_path)
        dst_dir_entry = self._get_dir_entry(dst_dir)
        if dst_dir_entry is None or (not create and dst_name not in dst_dir_entry):
            raise errors.ResourceNotFound(dst_path)
        dst_dir_entry.set_entry(dst_name, src_entry)
        src_dir_entry.remove_entry(src_name)
        src_entry.name = dst_name
        if preserve_time:
            copy_modified_time(self, src_path, self, dst_path)