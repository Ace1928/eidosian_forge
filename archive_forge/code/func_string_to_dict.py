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
def string_to_dict(string: str, pattern: str) -> Dict[str, str]:
    """Un-format a string using a python f-string pattern.
    From https://stackoverflow.com/a/36838374

    Example::

        >>> p = 'hello, my name is {name} and I am a {age} year old {what}'
        >>> s = p.format(name='cody', age=18, what='quarterback')
        >>> s
        'hello, my name is cody and I am a 18 year old quarterback'
        >>> string_to_dict(s, p)
        {'age': '18', 'name': 'cody', 'what': 'quarterback'}

    Args:
        string (str): input string
        pattern (str): pattern formatted like a python f-string

    Returns:
        Dict[str, str]: dictionary of variable -> value, retrieved from the input using the pattern

    Raises:
        ValueError: if the string doesn't match the pattern
    """
    regex = re.sub('{(.+?)}', '(?P<_\\1>.+)', pattern)
    result = re.search(regex, string)
    if result is None:
        raise ValueError(f"String {string} doesn't match the pattern {pattern}")
    values = list(result.groups())
    keys = re.findall('{(.+?)}', pattern)
    _dict = dict(zip(keys, values))
    return _dict