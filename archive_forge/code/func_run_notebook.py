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
def run_notebook(filename, opts, resources=None):
    """Loads and runs a notebook, returning both the version prior to
    running it and the version after running it.

    """
    with open(filename) as f:
        input_nb = nbformat.read(f, 4)
    cleaned_input_nb = copy.deepcopy(input_nb)
    for cell in cleaned_input_nb.cells:
        if 'execution_count' in cell:
            del cell['execution_count']
        cell['outputs'] = []
    if resources:
        opts = {'resources': resources, **opts}
    executor = NotebookClient(cleaned_input_nb, **opts)
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        with modified_env({'COLUMNS': '80', 'LINES': '24'}):
            output_nb = executor.execute()
    return (input_nb, output_nb)