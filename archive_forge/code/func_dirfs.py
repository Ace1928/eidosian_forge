import array
import logging
import posixpath
import warnings
from collections.abc import MutableMapping
from functools import cached_property
from fsspec.core import url_to_fs
@cached_property
def dirfs(self):
    """dirfs instance that can be used with the same keys as the mapper"""
    from .implementations.dirfs import DirFileSystem
    return DirFileSystem(path=self._root_key_to_str, fs=self.fs)