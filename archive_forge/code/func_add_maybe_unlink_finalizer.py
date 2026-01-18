from mmap import mmap
import errno
import os
import stat
import threading
import atexit
import tempfile
import time
import warnings
import weakref
from uuid import uuid4
from multiprocessing import util
from pickle import whichmodule, loads, dumps, HIGHEST_PROTOCOL, PicklingError
from .numpy_pickle import dump, load, load_temporary_memmap
from .backports import make_memmap
from .disk import delete_folder
from .externals.loky.backend import resource_tracker
def add_maybe_unlink_finalizer(memmap):
    util.debug('[FINALIZER ADD] adding finalizer to {} (id {}, filename {}, pid  {})'.format(type(memmap), id(memmap), os.path.basename(memmap.filename), os.getpid()))
    weakref.finalize(memmap, _log_and_unlink, memmap.filename)