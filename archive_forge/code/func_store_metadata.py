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
def store_metadata(self, path, metadata):
    """Store metadata of a computation."""
    try:
        item_path = os.path.join(self.location, *path)
        self.create_location(item_path)
        filename = os.path.join(item_path, 'metadata.json')

        def write_func(to_write, dest_filename):
            with self._open_item(dest_filename, 'wb') as f:
                f.write(json.dumps(to_write).encode('utf-8'))
        self._concurrency_safe_write(metadata, filename, write_func)
    except:
        pass