import atexit
import os
import shutil
import tempfile
import weakref
from fastimport.reftracker import RefTracker
from ... import lru_cache, trace
from . import branch_mapper
from .helpers import single_plural
def exit_cleanup():
    small_blob = small_blob_ref()
    if small_blob is not None:
        small_blob.close()
    shutil.rmtree(tempdir, ignore_errors=True)