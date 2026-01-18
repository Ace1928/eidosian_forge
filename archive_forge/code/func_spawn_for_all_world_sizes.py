import contextlib
import functools
import gc
import inspect
import logging
import multiprocessing
import os
import random
from statistics import mean
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import numpy
import pytest
import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import rpc
import torch.multiprocessing as mp
import torch.nn as nn
from fairscale.internal import torch_version
from fairscale.nn.model_parallel import destroy_model_parallel, initialize_model_parallel
from fairscale.nn.model_parallel.random import model_parallel_cuda_manual_seed
def spawn_for_all_world_sizes(test_func: Callable, world_sizes: List[int]=get_world_sizes(), args: Any=[], deterministic: bool=False) -> None:
    for world_size in world_sizes:
        _, filename = tempfile.mkstemp()
        _, filename_rpc = tempfile.mkstemp()
        try:
            mp.spawn(test_runner, args=(test_func, deterministic, world_size, filename, filename_rpc, *args), nprocs=world_size, join=True)
        finally:
            rmf(filename)
            rmf(filename_rpc)