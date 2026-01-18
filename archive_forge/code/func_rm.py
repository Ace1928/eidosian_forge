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
def rm(self, path, recursive=False, maxdepth=None):
    if not isinstance(path, list):
        path = [path]
    for p in path:
        p = self._strip_protocol(p).rstrip('/')
        if self.isdir(p):
            if not recursive:
                raise ValueError('Cannot delete directory, set recursive=True')
            if osp.abspath(p) == os.getcwd():
                raise ValueError('Cannot delete current working directory')
            shutil.rmtree(p)
        else:
            os.remove(p)