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
def test_execution_timing():
    """Compare the execution timing information stored in the cell with the
    actual time it took to run the cell. Also check for the cell timing string
    format."""
    opts = {'kernel_name': 'python'}
    input_name = 'Sleep1s.ipynb'
    input_file = os.path.join(current_dir, 'files', input_name)
    res = notebook_resources()
    input_nb, output_nb = run_notebook(input_file, opts, res)

    def get_time_from_str(s):
        time_format = '%Y-%m-%dT%H:%M:%S.%fZ'
        return datetime.datetime.strptime(s, time_format)
    execution_timing = output_nb['cells'][1]['metadata']['execution']
    status_busy = get_time_from_str(execution_timing['iopub.status.busy'])
    execute_input = get_time_from_str(execution_timing['iopub.execute_input'])
    execute_reply = get_time_from_str(execution_timing['shell.execute_reply'])
    status_idle = get_time_from_str(execution_timing['iopub.status.idle'])
    cell_start = get_time_from_str(output_nb['cells'][2]['outputs'][0]['text'])
    cell_end = get_time_from_str(output_nb['cells'][3]['outputs'][0]['text'])
    delta = datetime.timedelta(milliseconds=100)
    assert status_busy - cell_start < delta
    assert execute_input - cell_start < delta
    assert execute_reply - cell_end < delta
    assert status_idle - cell_end < delta