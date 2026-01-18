from __future__ import unicode_literals
import typing
import six
from . import errors
from .base import FS
from .copy import copy_dir, copy_file
from .error_tools import unwrap_errors
from .info import Info
from .path import abspath, join, normpath
def settimes(self, path, accessed=None, modified=None):
    self.check()
    _fs, _path = self.delegate_path(path)
    with unwrap_errors(path):
        _fs.settimes(_path, accessed=accessed, modified=modified)