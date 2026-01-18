from __future__ import annotations
import inspect
import logging
import os
import shutil
import uuid
from typing import Optional
from .asyn import AsyncFileSystem, _run_coros_in_chunks, sync_wrapper
from .callbacks import DEFAULT_CALLBACK
from .core import filesystem, get_filesystem_class, split_protocol, url_to_fs
def set_generic_fs(protocol, **storage_options):
    _generic_fs[protocol] = filesystem(protocol, **storage_options)