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
def test_bad_var_expand(self):
    """var_expand on invalid formats shouldn't raise"""
    self.assertEqual(ip.var_expand(u"{'a':5}"), u"{'a':5}")
    self.assertEqual(ip.var_expand(u'{asdf}'), u'{asdf}')
    self.assertEqual(ip.var_expand(u'{1/0}'), u'{1/0}')