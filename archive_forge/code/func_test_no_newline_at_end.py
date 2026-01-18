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
def test_no_newline_at_end(self):
    rows = b'{"a": 1,"b": 2, "c": 3}\n{"a": 4,"b": 5, "c": 6}'
    table = self.read_bytes(rows)
    assert table.to_pydict() == {'a': [1, 4], 'b': [2, 5], 'c': [3, 6]}