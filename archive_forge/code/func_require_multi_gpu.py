import asyncio
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import List, Union
from unittest import mock
import torch
import accelerate
from ..state import AcceleratorState, PartialState
from ..utils import (
def require_multi_gpu(test_case):
    """
    Decorator marking a test that requires a multi-GPU setup. These tests are skipped on a machine without multiple
    GPUs.
    """
    return unittest.skipUnless(torch.cuda.device_count() > 1, 'test requires multiple GPUs')(test_case)