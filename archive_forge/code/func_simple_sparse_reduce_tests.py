import faulthandler
import logging
import multiprocessing
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from functools import partial, reduce, wraps
from io import StringIO
from typing import Dict, NamedTuple, Optional, Union
from unittest.mock import patch
import torch
import torch._dynamo.test_case
import torch.cuda.nccl
import torch.distributed as c10d
import torch.nn as nn
from torch.testing._internal.common_utils import (
from torch.testing._internal.distributed.multi_threaded_pg import (
def simple_sparse_reduce_tests(rank: int, world_size: int, num_inputs: int=1):
    """
    Generate a number of basic test cases for sparse reduction.
    These cover tensors with a varying number of sparse dimensions and a varying
    number of dense dimensions. The only reduction operation we support is sum.
    """

    def generate(rank: int, world_size: int, sparse_dims: int=1, dense_dims: int=0):
        indices = torch.reshape(torch.arange(rank + 1), (1, rank + 1))
        shape = [world_size] + [2 for _ in range(dense_dims)]
        for _ in range(sparse_dims - 1):
            indices = torch.cat((indices, torch.zeros(1, rank + 1)))
            shape.append(world_size)
        values = torch.ones([rank + 1] + [2 for _ in range(dense_dims)])
        return torch.sparse_coo_tensor(indices, values, shape)

    def compute_sum(fn, world_size: int):
        return reduce(lambda a, b: a + b, [fn(rank, world_size) for rank in range(world_size)])
    return [([fn(num_inputs * rank + i, num_inputs * world_size) for i in range(num_inputs)], [compute_sum(fn, num_inputs * world_size) for i in range(num_inputs)]) for fn in [partial(generate, sparse_dims=1), partial(generate, sparse_dims=2), partial(generate, sparse_dims=3), partial(generate, dense_dims=1), partial(generate, dense_dims=2), partial(generate, dense_dims=3)]]