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
def test_block_overlapping_deprecation():

    @deprecated(fix="Don't use g.", deadline='v1000.0')
    def g(y):
        return y - 4

    @deprecated(fix="Don't use f.", deadline='v1000.0')
    def f(x):
        with block_overlapping_deprecation('g'):
            return [g(i + 1) for i in range(x)]
    with cirq.testing.assert_deprecated('f', deadline='v1000.0', count=1):
        f(5)