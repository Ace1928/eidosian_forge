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
def test_empty_rows(self):
    rows = b'{}\n{}\n'
    table = self.read_bytes(rows)
    schema = pa.schema([])
    assert table.schema == schema
    assert table.num_columns == 0
    assert table.num_rows == 2