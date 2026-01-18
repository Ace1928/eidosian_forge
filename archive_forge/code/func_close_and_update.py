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
def close_and_update(self, f, close):
    """Called when a file is closing, so store the set of blocks"""
    if f.closed:
        return
    path = self._strip_protocol(f.path)
    self._metadata.on_close_cached_file(f, path)
    try:
        logger.debug('going to save')
        self.save_cache()
        logger.debug('saved')
    except OSError:
        logger.debug('Cache saving failed while closing file')
    except NameError:
        logger.debug('Cache save failed due to interpreter shutdown')
    close()
    f.closed = True