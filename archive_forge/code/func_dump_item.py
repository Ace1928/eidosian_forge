from pickle import PicklingError
import re
import os
import os.path
import datetime
import json
import shutil
import warnings
import collections
import operator
import threading
from abc import ABCMeta, abstractmethod
from .backports import concurrency_safe_rename
from .disk import mkdirp, memstr_to_bytes, rm_subdirs
from . import numpy_pickle
def dump_item(self, path, item, verbose=1):
    """Dump an item in the store at the path given as a list of
           strings."""
    try:
        item_path = os.path.join(self.location, *path)
        if not self._item_exists(item_path):
            self.create_location(item_path)
        filename = os.path.join(item_path, 'output.pkl')
        if verbose > 10:
            print('Persisting in %s' % item_path)

        def write_func(to_write, dest_filename):
            with self._open_item(dest_filename, 'wb') as f:
                try:
                    numpy_pickle.dump(to_write, f, compress=self.compress)
                except PicklingError as e:
                    warnings.warn(f'Unable to cache to disk: failed to pickle output. In version 1.5 this will raise an exception. Exception: {e}.', FutureWarning)
        self._concurrency_safe_write(item, filename, write_func)
    except Exception as e:
        warnings.warn(f'Unable to cache to disk. Possibly a race condition in the creation of the directory. Exception: {e}.', CacheWarning)