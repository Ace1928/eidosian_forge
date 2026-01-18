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
def test_collect_yaml_no_top_level_dict(tmp_path):
    (tmp_path / 'a.yaml').write_bytes(b'[1234]')
    with pytest.raises(ValueError) as rec:
        list(collect_yaml(paths=[tmp_path]))
    assert 'a.yaml' in str(rec.value)
    assert 'is malformed' in str(rec.value)
    assert 'must have a dict' in str(rec.value)