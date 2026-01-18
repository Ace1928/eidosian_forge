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
def require_single_device(test_case):
    """
    Decorator marking a test that requires a single device. These tests are skipped when there is no hardware
    accelerator available or number of devices is more than one.
    """
    return unittest.skipUnless(torch_device != 'cpu' and device_count == 1, 'test requires a hardware accelerator')(test_case)