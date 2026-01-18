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
def test_merge_None_to_dict():
    assert dask.config.merge({'a': None, 'c': 0}, {'a': {'b': 1}}) == {'a': {'b': 1}, 'c': 0}