import pytest
from io import StringIO
from pathlib import Path
import gzip
import numpy
from .._json_api import (
from .._json_api import write_gzip_json, json_dumps, is_json_serializable
from .._json_api import json_loads
from ..util import force_string
from .util import make_tempdir
@pytest.mark.parametrize('obj,expected', [(['a', 'b', 1, 2], True), ({'a': 'b', 'c': 123}, True), ('hello', True), (lambda x: x, False)])
def test_is_json_serializable(obj, expected):
    assert is_json_serializable(obj) == expected