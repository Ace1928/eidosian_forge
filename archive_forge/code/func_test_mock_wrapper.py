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
@functools.wraps(func)
def test_mock_wrapper(self):
    """
            This inner function wrapper populates the executor object with
            the fake kernel client. This client has its iopub and shell
            channels mocked so as to fake the setup handshake and return
            the messages passed into prepare_cell_mocks as the execute_cell loop
            processes them.
            """
    cell_mock = NotebookNode(source='"foo" = "bar"', metadata={}, cell_type='code', outputs=[])

    class NotebookClientWithParentID(NotebookClient):
        parent_id: str
    nb = nbformat.v4.new_notebook()
    executor = NotebookClientWithParentID(nb)
    executor.nb.cells = [cell_mock]
    message_mock = iopub_messages_mock()
    executor.kc = MagicMock(iopub_channel=MagicMock(get_msg=message_mock), shell_channel=MagicMock(get_msg=shell_channel_message_mock()), execute=MagicMock(return_value=parent_id), is_alive=MagicMock(return_value=make_future(True)))
    executor.parent_id = parent_id
    return func(self, executor, cell_mock, message_mock)