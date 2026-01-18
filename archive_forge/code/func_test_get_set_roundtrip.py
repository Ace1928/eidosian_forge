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
@pytest.mark.parametrize('key', ['custom_key', 'custom-key'])
def test_get_set_roundtrip(key):
    value = 123
    with dask.config.set({key: value}):
        assert dask.config.get('custom_key') == value
        assert dask.config.get('custom-key') == value