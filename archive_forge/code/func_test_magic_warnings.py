import asyncio
import ast
import os
import signal
import shutil
import sys
import tempfile
import unittest
import pytest
from unittest import mock
from os.path import join
from IPython.core.error import InputRejected
from IPython.core.inputtransformer import InputTransformer
from IPython.core import interactiveshell
from IPython.core.oinspect import OInfo
from IPython.testing.decorators import (
from IPython.testing import tools as tt
from IPython.utils.process import find_cmd
import warnings
import warnings
@pytest.mark.parametrize('magic_cmd', ['pip', 'conda', 'cd'])
def test_magic_warnings(magic_cmd):
    if sys.platform == 'win32':
        to_mock = 'os.system'
        expected_arg, expected_kwargs = (magic_cmd, dict())
    else:
        to_mock = 'subprocess.call'
        expected_arg, expected_kwargs = (magic_cmd, dict(shell=True, executable=os.environ.get('SHELL', None)))
    with mock.patch(to_mock, return_value=0) as mock_sub:
        with pytest.warns(Warning, match='You executed the system command'):
            ip.system_raw(magic_cmd)
        mock_sub.assert_called_once_with(expected_arg, **expected_kwargs)