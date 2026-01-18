import asyncio
import concurrent.futures
import copy
import datetime
import functools
import os
import re
import sys
import threading
import warnings
from base64 import b64decode, b64encode
from queue import Empty
from typing import Any
from unittest.mock import MagicMock, Mock
import nbformat
import pytest
import xmltodict
from flaky import flaky  # type:ignore
from jupyter_client import KernelClient, KernelManager
from jupyter_client._version import version_info
from jupyter_client.kernelspec import KernelSpecManager
from nbconvert.filters import strip_ansi
from nbformat import NotebookNode
from testpath import modified_env
from traitlets import TraitError
from nbclient import NotebookClient, execute
from nbclient.exceptions import CellExecutionError
from .base import NBClientTestsBase
@flaky
def test_kernel_death_after_timeout(self):
    """Check that an error is raised when the kernel is_alive is false after a cell timed out"""
    filename = os.path.join(current_dir, 'files', 'Interrupt.ipynb')
    with open(filename) as f:
        input_nb = nbformat.read(f, 4)
    res = self.build_resources()
    res['metadata']['path'] = os.path.dirname(filename)
    executor = NotebookClient(input_nb, timeout=1)
    with pytest.raises(TimeoutError):
        executor.execute()
    km = executor.create_kernel_manager()

    async def is_alive():
        return False
    km.is_alive = is_alive
    with pytest.raises((RuntimeError, TimeoutError)):
        input_nb, output_nb = executor.execute()