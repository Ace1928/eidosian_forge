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
def test_many_async_parallel_notebooks(capfd):
    """Ensure that when many IPython kernels are run in parallel, nothing awful happens.

    Specifically, many IPython kernels when run simultaneously would encounter errors
    due to using the same SQLite history database.
    """
    opts = {'kernel_name': 'python', 'timeout': 5}
    input_name = 'HelloWorld.ipynb'
    input_file = os.path.join(current_dir, 'files', input_name)
    res = NBClientTestsBase().build_resources()
    res['metadata']['path'] = os.path.join(current_dir, 'files')
    run_notebook(input_file, opts, res)

    async def run_tasks():
        tasks = [async_run_notebook(input_file, opts, res) for i in range(4)]
        await asyncio.gather(*tasks)
    asyncio.run(run_tasks())
    captured = capfd.readouterr()
    assert filter_messages_on_error_output(captured.err) == ''