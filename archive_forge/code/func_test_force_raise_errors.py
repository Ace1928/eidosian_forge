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
def test_force_raise_errors(self):
    """
        Check that conversion halts if the ``force_raise_errors`` traitlet on
        NotebookClient is set to True.
        """
    filename = os.path.join(current_dir, 'files', 'Skip Exceptions with Cell Tags.ipynb')
    res = self.build_resources()
    res['metadata']['path'] = os.path.dirname(filename)
    with pytest.raises(CellExecutionError) as exc:
        run_notebook(filename, {'force_raise_errors': True}, res)
    exc_str = strip_ansi(str(exc.value))
    assert 'Exception: message' in exc_str
    if not sys.platform.startswith('win'):
        assert '# üñîçø∂é' in exc_str
    assert 'stderr' in exc_str
    assert 'stdout' in exc_str
    assert 'hello\n' in exc_str
    assert 'errorred\n' in exc_str
    assert '\n'.join(['', '----- stdout -----', 'hello', '---']) in exc_str
    assert '\n'.join(['', '----- stderr -----', 'errorred', '---']) in exc_str