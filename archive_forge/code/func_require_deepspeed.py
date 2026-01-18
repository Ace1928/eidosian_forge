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
def require_deepspeed(test_case):
    """
    Decorator marking a test that requires DeepSpeed installed. These tests are skipped when DeepSpeed isn't installed
    """
    return unittest.skipUnless(is_deepspeed_available(), 'test requires DeepSpeed')(test_case)