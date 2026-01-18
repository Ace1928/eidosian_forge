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
def test_dataclass_repr_numpy() -> None:

    @dataclasses.dataclass
    class TestClass2:
        x: np.ndarray

        def __repr__(self):
            return dataclass_repr(self, namespace='cirq.testing')
    tc = TestClass2(np.ones(3))
    assert repr(tc) == "cirq.testing.TestClass2(x=np.array([1.0, 1.0, 1.0], dtype=np.dtype('float64')))"