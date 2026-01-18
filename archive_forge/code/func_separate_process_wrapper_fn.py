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
def separate_process_wrapper_fn(func: Callable[[], None], do_multi_processing: bool) -> Callable[[], None]:
    """
    This function wraps another function into its own separated process. In order to ensure accurate memory
    measurements it is important that the function is executed in a separate process

    Args:
        - `func`: (`callable`): function() -> ... generic function which will be executed in its own separate process
        - `do_multi_processing`: (`bool`) Whether to run function on separate process or not
    """

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
    if do_multi_processing:
        logger.info(f'Function {func} is executed in its own process...')
        return multi_process_func
    else:
        return func