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
def test_can_pickle(self):
    """Can we pickle objects defined interactively (GH-29)"""
    ip = get_ipython()
    ip.reset()
    ip.run_cell('class Mylist(list):\n    def __init__(self,x=[]):\n        list.__init__(self,x)')
    ip.run_cell('w=Mylist([1,2,3])')
    from pickle import dumps
    _main = sys.modules['__main__']
    sys.modules['__main__'] = ip.user_module
    try:
        res = dumps(ip.user_ns['w'])
    finally:
        sys.modules['__main__'] = _main
    self.assertTrue(isinstance(res, bytes))