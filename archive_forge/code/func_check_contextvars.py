from __future__ import annotations
import dataclasses
import datetime
import decimal
import operator
import pathlib
import pickle
import random
import subprocess
import sys
import textwrap
from enum import Enum, Flag, IntEnum, IntFlag
from typing import Union
import cloudpickle
import pytest
from tlz import compose, curry, partial
import dask
from dask.base import TokenizationError, normalize_token, tokenize
from dask.core import literal
from dask.utils import tmpfile
from dask.utils_test import import_or_none
@pytest.fixture(autouse=True)
def check_contextvars():
    """Test that tokenize() and normalize_token() properly clean up context
    variables at all times
    """
    from dask.base import _ensure_deterministic, _seen
    with pytest.raises(LookupError):
        _ensure_deterministic.get()
    with pytest.raises(LookupError):
        _seen.get()
    yield
    with pytest.raises(LookupError):
        _ensure_deterministic.get()
    with pytest.raises(LookupError):
        _seen.get()