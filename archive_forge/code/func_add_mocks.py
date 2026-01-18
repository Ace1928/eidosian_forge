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
def add_mocks(self, mocks: Union[mock.Mock, List[mock.Mock]]):
    """
        Add custom mocks for tests that should be repeated on each test. Should be called during
        `MockingTestCase.setUp`, after `super().setUp()`.

        Args:
            mocks (`mock.Mock` or list of `mock.Mock`):
                Mocks that should be added to the `TestCase` after `TestCase.setUpClass` has been run
        """
    self.mocks = mocks if isinstance(mocks, (tuple, list)) else [mocks]
    for m in self.mocks:
        m.start()
        self.addCleanup(m.stop)