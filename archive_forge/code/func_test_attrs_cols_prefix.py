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
def test_attrs_cols_prefix(parser, geom_df):
    expected = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<doc:data xmlns:doc="http://example.xom">\n  <doc:row doc:index="0" doc:shape="square" doc:degrees="360" doc:sides="4.0"/>\n  <doc:row doc:index="1" doc:shape="circle" doc:degrees="360"/>\n  <doc:row doc:index="2" doc:shape="triangle" doc:degrees="180" doc:sides="3.0"/>\n</doc:data>'
    output = geom_df.to_xml(attr_cols=['index', 'shape', 'degrees', 'sides'], namespaces={'doc': 'http://example.xom'}, prefix='doc', parser=parser)
    output = equalize_decl(output)
    assert output == expected