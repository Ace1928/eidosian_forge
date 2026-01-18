from __future__ import annotations
from io import (
import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.common import get_handle
from pandas.io.xml import read_xml
def test_file_output_bytes_read(xml_books, parser, from_file_expected):
    df_file = read_xml(xml_books, parser=parser)
    with tm.ensure_clean('test.xml') as path:
        df_file.to_xml(path, parser=parser)
        with open(path, 'rb') as f:
            output = f.read().decode('utf-8').strip()
        output = equalize_decl(output)
        assert output == from_file_expected