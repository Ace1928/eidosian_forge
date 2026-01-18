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
def check_same_model_params(model_a: torch.nn.Module, model_b: torch.nn.Module, message: str='') -> None:
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b, atol=0.001), f'Model parameters differ\n{p_a} {p_b}\n' + message
    for b_a, b_b in zip(model_a.buffers(), model_b.buffers()):
        assert torch.allclose(b_a, b_b), f'Model buffers differ {b_a} - {b_b}\n' + message