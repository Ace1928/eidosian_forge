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
def require_single_xpu(test_case):
    """
    Decorator marking a test that requires CUDA on a single XPU. These tests are skipped when there are no XPU
    available or number of xPUs is more than one.
    """
    return unittest.skipUnless(torch.xpu.device_count() == 1, 'test requires a XPU')(test_case)