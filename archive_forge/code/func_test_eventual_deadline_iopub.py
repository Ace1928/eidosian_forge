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
@prepare_cell_mocks({'msg_type': 'stream', 'header': {'msg_type': 'stream'}, 'content': {'name': 'stdout', 'text': 'foo'}}, {'msg_type': 'stream', 'header': {'msg_type': 'stream'}, 'content': {'name': 'stderr', 'text': 'bar'}})
def test_eventual_deadline_iopub(self, executor, cell_mock, message_mock):

    def message_seq(messages):
        yield from messages
        while True:
            yield Empty()
    message_mock.side_effect = message_seq(list(message_mock.side_effect)[:-1])
    executor.kc.shell_channel.get_msg = Mock(return_value=make_future({'parent_header': {'msg_id': executor.parent_id}}))
    executor.raise_on_iopub_timeout = True
    with pytest.raises(TimeoutError):
        executor.execute_cell(cell_mock, 0)
    assert message_mock.call_count >= 3
    self.assertListEqual(cell_mock.outputs, [{'output_type': 'stream', 'name': 'stdout', 'text': 'foo'}, {'output_type': 'stream', 'name': 'stderr', 'text': 'bar'}])