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
def normalize_output(output):
    """
    Normalizes outputs for comparison.
    """
    output = dict(output)
    if 'metadata' in output:
        del output['metadata']
    if 'text' in output:
        output['text'] = re.sub(addr_pat, '<HEXADDR>', output['text'])
    if 'text/plain' in output.get('data', {}):
        output['data']['text/plain'] = re.sub(addr_pat, '<HEXADDR>', output['data']['text/plain'])
    if 'application/vnd.jupyter.widget-view+json' in output.get('data', {}):
        output['data']['application/vnd.jupyter.widget-view+json']['model_id'] = '<MODEL_ID>'
    if 'image/svg+xml' in output.get('data', {}):
        output['data']['image/svg+xml'] = xmltodict.parse(output['data']['image/svg+xml'])
    for key, value in output.get('data', {}).items():
        if isinstance(value, str):
            output['data'][key] = normalize_base64(value)
    if 'traceback' in output:
        tb = []
        for line in output['traceback']:
            line = re.sub(ipython_input_pat, '<IPY-INPUT>', strip_ansi(line))
            line = re.sub(ipython8_input_pat, '<IPY-INPUT>', strip_ansi(line))
            tb.append(line)
        output['traceback'] = tb
    return output