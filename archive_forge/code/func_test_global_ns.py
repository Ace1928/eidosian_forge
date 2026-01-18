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
def test_global_ns(self):
    """Code in functions must be able to access variables outside them."""
    ip = get_ipython()
    ip.run_cell('a = 10')
    ip.run_cell('def f(x):\n    return x + a')
    ip.run_cell('b = f(12)')
    self.assertEqual(ip.user_ns['b'], 22)