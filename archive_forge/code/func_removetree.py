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
def removetree(self, path):
    _path = self.validatepath(path)
    with self._lock:
        if _path == '/':
            self.root.clear()
            return
        dir_path, file_name = split(_path)
        parent_dir_entry = self._get_dir_entry(dir_path)
        if parent_dir_entry is None or file_name not in parent_dir_entry:
            raise errors.ResourceNotFound(path)
        dir_dir_entry = typing.cast(_DirEntry, self._get_dir_entry(_path))
        if not dir_dir_entry.is_dir:
            raise errors.DirectoryExpected(path)
        parent_dir_entry.remove_entry(file_name)