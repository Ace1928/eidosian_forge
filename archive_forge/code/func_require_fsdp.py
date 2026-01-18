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
def require_fsdp(test_case):
    """
    Decorator marking a test that requires FSDP installed. These tests are skipped when FSDP isn't installed
    """
    return unittest.skipUnless(is_torch_version('>=', '1.12.0'), 'test requires torch version >= 1.12.0')(test_case)