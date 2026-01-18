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
def test_block_sizes(self):
    rows = b'{"a": 1}\n{"a": 2}\n{"a": 3}'
    read_options = ReadOptions()
    parse_options = ParseOptions()
    for data in [rows, rows + b'\n']:
        for newlines_in_values in [False, True]:
            parse_options.newlines_in_values = newlines_in_values
            read_options.block_size = 4
            with pytest.raises(ValueError, match='try to increase block size'):
                self.read_bytes(data, read_options=read_options, parse_options=parse_options)
            for block_size in range(9, 20):
                read_options.block_size = block_size
                table = self.read_bytes(data, read_options=read_options, parse_options=parse_options)
                assert table.to_pydict() == {'a': [1, 2, 3]}