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
def test_missing_prefix_in_nmsp(parser, geom_df):
    with pytest.raises(KeyError, match='doc is not included in namespaces'):
        geom_df.to_xml(namespaces={'': 'http://example.com'}, prefix='doc', parser=parser)