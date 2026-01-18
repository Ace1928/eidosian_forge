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
@pytest.mark.xfail('python3' not in KernelSpecManager().find_kernel_specs(), reason='requires a python3 kernelspec')
def test_empty_kernel_name(self):
    """Can kernel in nb metadata be found when an empty string is passed?

        Note: this pattern should be discouraged in practice.
        Passing in no kernel_name to NotebookClient is recommended instead.
        """
    filename = os.path.join(current_dir, 'files', 'UnicodePy3.ipynb')
    res = self.build_resources()
    input_nb, output_nb = run_notebook(filename, {'kernel_name': ''}, res)
    assert_notebooks_equal(input_nb, output_nb)
    with pytest.raises(TraitError):
        input_nb, output_nb = run_notebook(filename, {'kernel_name': None}, res)