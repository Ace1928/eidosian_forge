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
@prepare_cell_mocks(reply_msg={'msg_type': 'execute_reply', 'header': {'msg_type': 'execute_reply'}, 'content': {'status': 'error'}})
def test_no_source(self, executor, cell_mock, message_mock):
    cell_mock = NotebookNode(source='     ', metadata={}, cell_type='code', outputs=[])
    executor.execute_cell(cell_mock, 0)
    assert message_mock.call_count == 0
    assert cell_mock.outputs == []