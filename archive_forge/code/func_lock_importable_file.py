import copy
import functools
import itertools
import multiprocessing.pool
import os
import queue
import re
import types
import warnings
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from multiprocessing import Manager
from pathlib import Path
from queue import Empty
from shutil import disk_usage
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union
from urllib.parse import urlparse
import multiprocess
import multiprocess.pool
import numpy as np
from tqdm.auto import tqdm
from .. import config
from ..parallel import parallel_map
from . import logging
from . import tqdm as hf_tqdm
from ._dill import (  # noqa: F401 # imported for backward compatibility. TODO: remove in 3.0.0
from ._filelock import FileLock
def lock_importable_file(importable_local_file: str) -> FileLock:
    importable_directory_path = str(Path(importable_local_file).resolve().parent.parent)
    lock_path = importable_directory_path + '.lock'
    return FileLock(lock_path)