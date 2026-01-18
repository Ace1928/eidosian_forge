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
def get_executor_with_hooks(nb=None, executor=None, async_hooks=False):
    if async_hooks:
        hooks = {key: AsyncMock() for key in hook_methods}
    else:
        hooks = {key: MagicMock() for key in hook_methods}
    if nb is not None:
        if executor is not None:
            raise RuntimeError('Cannot pass nb and executor at the same time')
        executor = NotebookClient(nb)
    for k, v in hooks.items():
        setattr(executor, k, v)
    return (executor, hooks)