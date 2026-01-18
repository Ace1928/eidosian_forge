from collections import OrderedDict
from decimal import Decimal
import io
import itertools
import json
import string
import unittest
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.json import read_json, ReadOptions, ParseOptions
def test_explicit_schema_with_unexpected_behaviour(self):
    rows = b'{"foo": "bar", "num": 0}\n{"foo": "baz", "num": 1}\n'
    schema = pa.schema([('foo', pa.binary())])
    opts = ParseOptions(explicit_schema=schema)
    table = self.read_bytes(rows, parse_options=opts)
    assert table.schema == pa.schema([('foo', pa.binary()), ('num', pa.int64())])
    assert table.to_pydict() == {'foo': [b'bar', b'baz'], 'num': [0, 1]}
    opts = ParseOptions(explicit_schema=schema, unexpected_field_behavior='ignore')
    table = self.read_bytes(rows, parse_options=opts)
    assert table.schema == pa.schema([('foo', pa.binary())])
    assert table.to_pydict() == {'foo': [b'bar', b'baz']}
    opts = ParseOptions(explicit_schema=schema, unexpected_field_behavior='error')
    with pytest.raises(pa.ArrowInvalid, match='JSON parse error: unexpected field'):
        self.read_bytes(rows, parse_options=opts)