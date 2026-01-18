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
@pytest.mark.parametrize('inp,out', [('1', '1'), (1, 1), ('$FOO', 'foo'), ([1, '$FOO'], [1, 'foo']), ((1, '$FOO'), (1, 'foo')), ({1, '$FOO'}, {1, 'foo'}), ({'a': '$FOO'}, {'a': 'foo'}), ({'a': 'A', 'b': [1, '2', '$FOO']}, {'a': 'A', 'b': [1, '2', 'foo']})])
def test_expand_environment_variables(monkeypatch, inp, out):
    monkeypatch.setenv('FOO', 'foo')
    assert expand_environment_variables(inp) == out