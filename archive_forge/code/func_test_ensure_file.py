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
def test_ensure_file(tmp_path):
    a = {'x': 1, 'y': {'a': 1}}
    b = {'x': 123}
    source = tmp_path / 'source.yaml'
    dest = tmp_path / 'dest'
    destination = tmp_path / 'dest' / 'source.yaml'
    source.write_text(yaml.dump(a))
    ensure_file(source=source, destination=dest, comment=False)
    result = yaml.safe_load(destination.read_text())
    assert result == a
    source.write_text(yaml.dump(b))
    ensure_file(source=source, destination=dest, comment=False)
    result = yaml.safe_load(destination.read_text())
    assert result == a
    os.remove(destination)
    ensure_file(source=source, destination=dest, comment=True)
    text = destination.read_text()
    assert '123' in text
    result = yaml.safe_load(destination.read_text())
    assert not result