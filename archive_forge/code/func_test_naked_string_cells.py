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
def test_naked_string_cells(self):
    """Test that cells with only naked strings are fully executed"""
    ip.run_cell('"a"\n')
    self.assertEqual(ip.user_ns['_'], 'a')
    ip.run_cell('"""a\nb"""\n')
    self.assertEqual(ip.user_ns['_'], 'a\nb')