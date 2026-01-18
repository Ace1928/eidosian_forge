from __future__ import unicode_literals
import typing
import six
from . import errors
from .base import FS
from .copy import copy_dir, copy_file
from .error_tools import unwrap_errors
from .info import Info
from .path import abspath, join, normpath
def filterdir(self, path, files=None, dirs=None, exclude_dirs=None, exclude_files=None, namespaces=None, page=None):
    self.check()
    _fs, _path = self.delegate_path(path)
    iter_files = iter(_fs.filterdir(_path, exclude_dirs=exclude_dirs, exclude_files=exclude_files, files=files, dirs=dirs, namespaces=namespaces, page=page))
    with unwrap_errors(path):
        for info in iter_files:
            yield info