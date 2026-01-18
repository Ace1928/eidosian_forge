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
def test_custom_kernel_manager(self):
    from .fake_kernelmanager import FakeCustomKernelManager
    filename = os.path.join(current_dir, 'files', 'HelloWorld.ipynb')
    with open(filename) as f:
        input_nb = nbformat.read(f, 4)
    cleaned_input_nb = copy.deepcopy(input_nb)
    for cell in cleaned_input_nb.cells:
        if 'execution_count' in cell:
            del cell['execution_count']
        cell['outputs'] = []
    executor = NotebookClient(cleaned_input_nb, resources=self.build_resources(), kernel_manager_class=FakeCustomKernelManager)
    with modified_env({'COLUMNS': '80', 'LINES': '24'}):
        executor.execute()
    expected = FakeCustomKernelManager.expected_methods.items()
    for method, call_count in expected:
        self.assertNotEqual(call_count, 0, f'{method} was called')