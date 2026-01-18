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
def require_cpu(test_case):
    """
    Decorator marking a test that must be only ran on the CPU. These tests are skipped when a GPU is available.
    """
    return unittest.skipUnless(torch_device == 'cpu', 'test requires only a CPU')(test_case)