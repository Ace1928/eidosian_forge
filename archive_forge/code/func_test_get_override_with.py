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
def test_get_override_with():
    with dask.config.set({'foo': 'bar'}):
        assert dask.config.get('foo') == 'bar'
        assert dask.config.get('foo', override_with=None) == 'bar'
        assert dask.config.get('foo', override_with='baz') == 'baz'
        assert dask.config.get('foo', override_with=False) is False
        assert dask.config.get('foo', override_with=True) is True
        assert dask.config.get('foo', override_with=123) == 123
        assert dask.config.get('foo', override_with={'hello': 'world'}) == {'hello': 'world'}
        assert dask.config.get('foo', override_with=['one']) == ['one']