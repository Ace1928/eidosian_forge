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
def test_ensure_file_defaults_to_DASK_CONFIG_directory(tmp_path):
    a = {'x': 1, 'y': {'a': 1}}
    source = tmp_path / 'source.yaml'
    source.write_text(yaml.dump(a))
    destination = tmp_path / 'dask'
    PATH = dask.config.PATH
    try:
        dask.config.PATH = destination
        ensure_file(source=source)
    finally:
        dask.config.PATH = PATH
    assert destination.is_dir()
    [fn] = os.listdir(destination)
    assert os.path.split(fn)[1] == os.path.split(source)[1]