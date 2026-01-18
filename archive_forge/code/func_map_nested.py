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
def map_nested(function: Callable[[Any], Any], data_struct: Any, dict_only: bool=False, map_list: bool=True, map_tuple: bool=False, map_numpy: bool=False, num_proc: Optional[int]=None, parallel_min_length: int=2, types: Optional[tuple]=None, disable_tqdm: bool=True, desc: Optional[str]=None) -> Any:
    """Apply a function recursively to each element of a nested data struct.

    Use multiprocessing if num_proc > 1 and the length of data_struct is greater than or equal to
    `parallel_min_length`.

    <Changed version="2.5.0">

    Before version 2.5.0, multiprocessing was not used if `num_proc` was greater than or equal to ``len(iterable)``.

    Now, if `num_proc` is greater than or equal to ``len(iterable)``, `num_proc` is set to ``len(iterable)`` and
    multiprocessing is used.

    </Changed>

    Args:
        function (`Callable`): Function to be applied to `data_struct`.
        data_struct (`Any`): Data structure to apply `function` to.
        dict_only (`bool`, default `False`): Whether only apply `function` recursively to `dict` values in
            `data_struct`.
        map_list (`bool`, default `True`): Whether also apply `function` recursively to `list` elements (besides `dict`
            values).
        map_tuple (`bool`, default `False`): Whether also apply `function` recursively to `tuple` elements (besides
            `dict` values).
        map_numpy (`bool, default `False`): Whether also apply `function` recursively to `numpy.array` elements (besides
            `dict` values).
        num_proc (`int`, *optional*): Number of processes.
        parallel_min_length (`int`, default `2`): Minimum length of `data_struct` required for parallel
            processing.
            <Added version="2.5.0"/>
        types (`tuple`, *optional*): Additional types (besides `dict` values) to apply `function` recursively to their
            elements.
        disable_tqdm (`bool`, default `True`): Whether to disable the tqdm progressbar.
        desc (`str`, *optional*): Prefix for the tqdm progressbar.

    Returns:
        `Any`
    """
    if types is None:
        types = []
        if not dict_only:
            if map_list:
                types.append(list)
            if map_tuple:
                types.append(tuple)
            if map_numpy:
                types.append(np.ndarray)
        types = tuple(types)
    if not isinstance(data_struct, dict) and (not isinstance(data_struct, types)):
        return function(data_struct)
    iterable = list(data_struct.values()) if isinstance(data_struct, dict) else data_struct
    if num_proc is None:
        num_proc = 1
    if any((isinstance(v, types) and len(v) > len(iterable) for v in iterable)):
        mapped = [map_nested(function=function, data_struct=obj, num_proc=num_proc, parallel_min_length=parallel_min_length, types=types) for obj in iterable]
    elif num_proc != -1 and num_proc <= 1 or len(iterable) < parallel_min_length:
        mapped = [_single_map_nested((function, obj, types, None, True, None)) for obj in hf_tqdm(iterable, disable=disable_tqdm, desc=desc)]
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.* is experimental and might be subject to breaking changes in the future\\.$', category=UserWarning)
            mapped = parallel_map(function, iterable, num_proc, types, disable_tqdm, desc, _single_map_nested)
    if isinstance(data_struct, dict):
        return dict(zip(data_struct.keys(), mapped))
    elif isinstance(data_struct, list):
        return mapped
    elif isinstance(data_struct, tuple):
        return tuple(mapped)
    else:
        return np.array(mapped)