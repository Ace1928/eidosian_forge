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
def test_future_flags(self):
    """Check that future flags are used for parsing code (gh-777)"""
    ip.run_cell('from __future__ import barry_as_FLUFL')
    try:
        ip.run_cell('prfunc_return_val = 1 <> 2')
        assert 'prfunc_return_val' in ip.user_ns
    finally:
        ip.compile.reset_compiler_flags()