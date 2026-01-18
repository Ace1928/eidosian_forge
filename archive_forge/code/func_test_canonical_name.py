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
def test_canonical_name():
    c = {'foo-bar': 1, 'fizz_buzz': 2}
    assert canonical_name('foo-bar', c) == 'foo-bar'
    assert canonical_name('foo_bar', c) == 'foo-bar'
    assert canonical_name('fizz-buzz', c) == 'fizz_buzz'
    assert canonical_name('fizz_buzz', c) == 'fizz_buzz'
    assert canonical_name('new-key', c) == 'new-key'
    assert canonical_name('new_key', c) == 'new_key'