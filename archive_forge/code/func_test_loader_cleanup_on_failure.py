import collections
import dataclasses
import importlib.metadata
import inspect
import logging
import multiprocessing
import os
import sys
import traceback
import types
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Tuple
from importlib.machinery import ModuleSpec
from unittest import mock
import duet
import numpy as np
import pandas as pd
import pytest
import sympy
from _pytest.outcomes import Failed
import cirq.testing
from cirq._compat import (
def test_loader_cleanup_on_failure():

    class FakeLoader(importlib.abc.Loader):

        def exec_module(self, module: ModuleType) -> None:
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        module = types.ModuleType('old')
        DeprecatedModuleLoader(FakeLoader(), 'old', 'new').exec_module(module)
    assert 'old' not in sys.modules
    assert 'new' not in sys.modules