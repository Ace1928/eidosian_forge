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
def test_custom_exc_count():
    hook = mock.Mock(return_value=None)
    ip.set_custom_exc((SyntaxError,), hook)
    before = ip.execution_count
    ip.run_cell('def foo()', store_history=True)
    ip.set_custom_exc((), None)
    assert hook.call_count == 1
    assert ip.execution_count == before + 1