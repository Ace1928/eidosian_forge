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
def register_new_context(self, context_id):
    if context_id in self._cached_temp_folders:
        return
    else:
        new_folder_name = 'joblib_memmapping_folder_{}_{}_{}'.format(os.getpid(), self._id, context_id)
        new_folder_path, _ = _get_temp_dir(new_folder_name, self._temp_folder_root)
        self.register_folder_finalizer(new_folder_path, context_id)
        self._cached_temp_folders[context_id] = new_folder_path