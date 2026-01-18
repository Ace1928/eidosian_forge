from __future__ import annotations
import os
import pathlib
import site
import stat
import sys
from collections import OrderedDict
from contextlib import contextmanager
import pytest
import yaml
import dask.config
from dask.config import (
def test_env_var_canonical_name(monkeypatch):
    value = 3
    monkeypatch.setenv('DASK_A_B', str(value))
    d = {}
    dask.config.refresh(config=d)
    assert get('a_b', config=d) == value
    assert get('a-b', config=d) == value