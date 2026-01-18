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
def test_dont_cache_with_semicolon(self):
    """Ending a line with semicolon should not cache the returned object (GH-307)"""
    oldlen = len(ip.user_ns['Out'])
    for cell in ['1;', '1;1;']:
        res = ip.run_cell(cell, store_history=True)
        newlen = len(ip.user_ns['Out'])
        self.assertEqual(oldlen, newlen)
        self.assertIsNone(res.result)
    i = 0
    for cell in ['1', '1;1']:
        ip.run_cell(cell, store_history=True)
        newlen = len(ip.user_ns['Out'])
        i += 1
        self.assertEqual(oldlen + i, newlen)