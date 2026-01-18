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
def subprocess_context(test_func):
    """Ensures that sys.modules changes in subprocesses won't impact the parent process."""
    assert callable(test_func), 'subprocess_context expects a function. Did you call the function instead of passing it to this method?'
    ctx = multiprocessing.get_context('spawn' if os.name == 'nt' else 'fork')
    exception = ctx.Queue()

    def isolated_func(*args, **kwargs):
        kwargs['queue'] = exception
        kwargs['func'] = test_func
        p = ctx.Process(target=_trace_unhandled_exceptions, args=args, kwargs=kwargs)
        p.start()
        p.join()
        result = exception.get()
        if result:
            ex_type, msg, ex_trace = result
            if ex_type == 'Skipped':
                warnings.warn(f'Skipping: {ex_type}: {msg}\n{ex_trace}')
                pytest.skip(f'{ex_type}: {msg}\n{ex_trace}')
            else:
                pytest.fail(f'{ex_type}: {msg}\n{ex_trace}')
    return isolated_func