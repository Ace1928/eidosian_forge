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
def require_wandb(test_case):
    """
    Decorator marking a test that requires wandb installed. These tests are skipped when wandb isn't installed
    """
    return unittest.skipUnless(is_wandb_available(), 'test requires wandb')(test_case)