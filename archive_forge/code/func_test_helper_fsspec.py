from __future__ import absolute_import, print_function, division
import sys
import os
from importlib import import_module
import pytest
from petl.compat import PY3
from petl.test.helpers import ieq, eq_
from petl.io.avro import fromavro, toavro
from petl.io.csv import fromcsv, tocsv
from petl.io.json import fromjson, tojson
from petl.io.xlsx import fromxlsx, toxlsx
from petl.io.xls import fromxls, toxls
from petl.util.vis import look
def test_helper_fsspec():
    try:
        import fsspec
    except ImportError as e:
        pytest.skip('SKIP FSSPEC helper tests: %s' % e)
    else:
        _write_read_from_env_matching('PETL_TEST_')