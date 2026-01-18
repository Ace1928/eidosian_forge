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
def skip_if_no_gpu(func):
    """Skips if the world size exceeds the number of GPUs, ensuring that if the
    test is run, each rank has its own GPU via ``torch.cuda.device(rank)``."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            sys.exit(TEST_SKIPS['no_cuda'].exit_code)
        world_size = int(os.environ['WORLD_SIZE'])
        if torch.cuda.device_count() < world_size:
            sys.exit(TEST_SKIPS[f'multi-gpu-{world_size}'].exit_code)
        return func(*args, **kwargs)
    return wrapper