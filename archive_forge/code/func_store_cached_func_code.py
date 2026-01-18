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
def store_cached_func_code(self, path, func_code=None):
    """Store the code of the cached function."""
    func_path = os.path.join(self.location, *path)
    if not self._item_exists(func_path):
        self.create_location(func_path)
    if func_code is not None:
        filename = os.path.join(func_path, 'func_code.py')
        with self._open_item(filename, 'wb') as f:
            f.write(func_code.encode('utf-8'))