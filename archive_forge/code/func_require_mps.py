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
def require_mps(test_case):
    """
    Decorator marking a test that requires MPS backend. These tests are skipped when torch doesn't support `mps`
    backend.
    """
    return unittest.skipUnless(is_mps_available(), 'test requires a `mps` backend support in `torch`')(test_case)