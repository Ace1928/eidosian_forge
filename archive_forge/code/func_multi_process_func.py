import copy
import csv
import linecache
import os
import platform
import sys
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from datetime import datetime
from multiprocessing import Pipe, Process, Queue
from multiprocessing.connection import Connection
from typing import Callable, Iterable, List, NamedTuple, Optional, Union
from .. import AutoConfig, PretrainedConfig
from .. import __version__ as version
from ..utils import is_psutil_available, is_py3nvml_available, is_tf_available, is_torch_available, logging
from .benchmark_args_utils import BenchmarkArguments
def multi_process_func(*args, **kwargs):

    def wrapper_func(queue: Queue, *args):
        try:
            result = func(*args)
        except Exception as e:
            logger.error(e)
            print(e)
            result = 'N/A'
        queue.put(result)
    queue = Queue()
    p = Process(target=wrapper_func, args=[queue] + list(args))
    p.start()
    result = queue.get()
    p.join()
    return result