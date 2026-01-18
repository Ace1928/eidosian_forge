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
def contains_item(self, path):
    """Check if there is an item at the path, given as a list of
           strings"""
    item_path = os.path.join(self.location, *path)
    filename = os.path.join(item_path, 'output.pkl')
    return self._item_exists(filename)