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
def test_bad_custom_tb_return(self):
    """Check that InteractiveShell is protected from bad return types in custom exception handlers"""
    ip.set_custom_exc((NameError,), lambda etype, value, tb, tb_offset=None: 1)
    self.assertEqual(ip.custom_exceptions, (NameError,))
    with tt.AssertPrints('Custom TB Handler failed', channel='stderr'):
        ip.run_cell(u'a=abracadabra')
    self.assertEqual(ip.custom_exceptions, ())