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
def print_memory_trace_statistics(self, summary: MemorySummary):
    self.print_fn('\nLine by line memory consumption:\n' + '\n'.join((f'{state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}' for state in summary.sequential)))
    self.print_fn('\nLines with top memory consumption:\n' + '\n'.join((f'=> {state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}' for state in summary.cumulative[:6])))
    self.print_fn('\nLines with lowest memory consumption:\n' + '\n'.join((f'=> {state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}' for state in summary.cumulative[-6:])))
    self.print_fn(f'\nTotal memory increase: {summary.total}')