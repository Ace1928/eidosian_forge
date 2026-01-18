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
def concurrency_safe_write(object_to_write, filename, write_func):
    """Writes an object into a unique file in a concurrency-safe way."""
    thread_id = id(threading.current_thread())
    temporary_filename = '{}.thread-{}-pid-{}'.format(filename, thread_id, os.getpid())
    write_func(object_to_write, temporary_filename)
    return temporary_filename