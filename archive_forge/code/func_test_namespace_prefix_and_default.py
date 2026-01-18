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
def test_namespace_prefix_and_default(parser, geom_df):
    expected = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<doc:data xmlns:doc="http://other.org" xmlns="http://example.com">\n  <doc:row>\n    <doc:index>0</doc:index>\n    <doc:shape>square</doc:shape>\n    <doc:degrees>360</doc:degrees>\n    <doc:sides>4.0</doc:sides>\n  </doc:row>\n  <doc:row>\n    <doc:index>1</doc:index>\n    <doc:shape>circle</doc:shape>\n    <doc:degrees>360</doc:degrees>\n    <doc:sides/>\n  </doc:row>\n  <doc:row>\n    <doc:index>2</doc:index>\n    <doc:shape>triangle</doc:shape>\n    <doc:degrees>180</doc:degrees>\n    <doc:sides>3.0</doc:sides>\n  </doc:row>\n</doc:data>'
    output = geom_df.to_xml(namespaces={'': 'http://example.com', 'doc': 'http://other.org'}, prefix='doc', parser=parser)
    output = equalize_decl(output)
    assert output == expected