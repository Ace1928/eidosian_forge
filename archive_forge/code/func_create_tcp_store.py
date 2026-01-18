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
@retry_on_connect_failures
def create_tcp_store(addr='localhost', world_size=1, is_master=True, timeout=timedelta(minutes=5), wait_for_workers=True, jit_class=False, use_libuv=False):
    """
    Creates a TCP store. Retries if the chosen port is already in use.
    """
    port = find_free_port()
    if jit_class:
        timeout_millisecond = int(timeout / timedelta(milliseconds=1))
        return torch.classes.dist_c10d.TCPStore(addr, port, world_size, is_master, timeout_millisecond)
    else:
        return c10d.TCPStore(addr, port, world_size, is_master, wait_for_workers=wait_for_workers, use_libuv=use_libuv)