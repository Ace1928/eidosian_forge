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
def test_attrs_cols_nan_output(parser, geom_df):
    expected = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<data>\n  <row index="0" shape="square" degrees="360" sides="4.0"/>\n  <row index="1" shape="circle" degrees="360"/>\n  <row index="2" shape="triangle" degrees="180" sides="3.0"/>\n</data>'
    output = geom_df.to_xml(attr_cols=['shape', 'degrees', 'sides'], parser=parser)
    output = equalize_decl(output)
    assert output == expected