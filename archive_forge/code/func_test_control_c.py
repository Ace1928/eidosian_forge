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
@mock.patch('subprocess.call', side_effect=KeyboardInterrupt)
@mock.patch('os.system', side_effect=KeyboardInterrupt)
def test_control_c(self, *mocks):
    try:
        self.system('sleep 1 # wont happen')
    except KeyboardInterrupt:
        self.fail('system call should intercept keyboard interrupt from subprocess.call')
    self.assertEqual(ip.user_ns['_exit_code'], -signal.SIGINT)