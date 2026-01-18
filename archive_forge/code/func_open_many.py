from __future__ import annotations
import inspect
import logging
import os
import tempfile
import time
import weakref
from shutil import rmtree
from typing import TYPE_CHECKING, Any, Callable, ClassVar
from fsspec import AbstractFileSystem, filesystem
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.compression import compr
from fsspec.core import BaseCache, MMapCache
from fsspec.exceptions import BlocksizeMismatchError
from fsspec.implementations.cache_mapper import create_cache_mapper
from fsspec.implementations.cache_metadata import CacheMetadata
from fsspec.spec import AbstractBufferedFile
from fsspec.transaction import Transaction
from fsspec.utils import infer_compression
def open_many(self, open_files, **kwargs):
    paths = [of.path for of in open_files]
    if 'r' in open_files.mode:
        self._mkcache()
    else:
        return [LocalTempFile(self.fs, path, mode=open_files.mode, fn=os.path.join(self.storage[-1], self._mapper(path)), **kwargs) for path in paths]
    if self.compression:
        raise NotImplementedError
    details = [self._check_file(sp) for sp in paths]
    downpath = [p for p, d in zip(paths, details) if not d]
    downfn0 = [os.path.join(self.storage[-1], self._mapper(p)) for p, d in zip(paths, details)]
    downfn = [fn for fn, d in zip(downfn0, details) if not d]
    if downpath:
        self.fs.get(downpath, downfn)
        newdetail = [{'original': path, 'fn': self._mapper(path), 'blocks': True, 'time': time.time(), 'uid': self.fs.ukey(path)} for path in downpath]
        for path, detail in zip(downpath, newdetail):
            self._metadata.update_file(path, detail)
        self.save_cache()

    def firstpart(fn):
        return fn[1] if isinstance(fn, tuple) else fn
    return [open(firstpart(fn0) if fn0 else fn1, mode=open_files.mode) for fn0, fn1 in zip(details, downfn0)]