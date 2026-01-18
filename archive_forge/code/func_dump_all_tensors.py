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
def dump_all_tensors(rank: int) -> None:
    """Useful tool for debugging memory issues from the python side."""
    if rank != 0:
        return
    for obj in gc.get_objects():
        try:
            ttype = str(type(obj))
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(ttype, obj.shape, obj.dtype, obj.device, obj.storage().size())
        except Exception:
            pass
    print(torch.cuda.memory_summary())