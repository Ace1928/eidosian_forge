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
def test_deprecated_with_property():

    class AClass(object):

        def __init__(self, a):
            self.a = a

        @property
        @deprecated(deadline='v1.2', fix='Stop using.', name='AClass.test_func')
        def f(self):
            return self.a
    instance = AClass(4)
    with cirq.testing.assert_deprecated('_compat_test.py:', 'AClass.test_func was used', 'will be removed in cirq v1.2', 'Stop using.', deadline='v1.2'):
        assert instance.f == 4