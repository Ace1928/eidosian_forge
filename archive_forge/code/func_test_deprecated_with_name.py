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
def test_deprecated_with_name():

    @deprecated(deadline='v1.2', fix='Roll some dice.', name='test_func')
    def f(a, b):
        return a + b
    with cirq.testing.assert_deprecated('_compat_test.py:', 'test_func was used', 'will be removed in cirq v1.2', 'Roll some dice.', deadline='v1.2'):
        assert f(1, 2) == 3