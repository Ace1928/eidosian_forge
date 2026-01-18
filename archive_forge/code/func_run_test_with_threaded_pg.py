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
def run_test_with_threaded_pg(self, test_name, rank, world_size):
    """
        Run the current test associated with `test_name` using the threaded process group.
        """
    c10d.init_process_group(backend='threaded', rank=rank, world_size=world_size, store=self.__class__.global_store)
    self.perThreadSetUp()
    try:
        getattr(self, test_name)()
    except BaseException as ex:
        self.exception_queue.put((rank, sys.exc_info()))
        ProcessLocalGroup.exception_handle(ex)
    finally:
        c10d.destroy_process_group()
        self.perThreadTearDown()