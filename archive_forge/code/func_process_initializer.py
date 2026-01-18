import os
import multiprocessing as mp
from multiprocessing import Pool, cpu_count, pool
from traceback import format_exception
import sys
from logging import INFO
import gc
from copy import deepcopy
import numpy as np
from ... import logging
from ...utils.profiler import get_system_total_memory_gb
from ..engine import MapNode
from .base import DistributedPluginBase
def process_initializer(cwd):
    """Initializes the environment of the child process"""
    os.chdir(cwd)
    os.environ['NIPYPE_NO_ET'] = '1'