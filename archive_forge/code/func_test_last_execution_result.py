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
def test_last_execution_result(self):
    """ Check that last execution result gets set correctly (GH-10702) """
    result = ip.run_cell('a = 5; a')
    self.assertTrue(ip.last_execution_succeeded)
    self.assertEqual(ip.last_execution_result.result, 5)
    result = ip.run_cell('a = x_invalid_id_x')
    self.assertFalse(ip.last_execution_succeeded)
    self.assertFalse(ip.last_execution_result.success)
    self.assertIsInstance(ip.last_execution_result.error_in_exec, NameError)