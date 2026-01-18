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
def contains_path(self, path):
    """Check cached function is available in store."""
    func_path = os.path.join(self.location, *path)
    return self.object_exists(func_path)