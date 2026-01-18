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
def test_na_empty_str_elem_option(parser, geom_df):
    output = geom_df.to_xml(na_rep='', parser=parser)
    output = equalize_decl(output)
    assert output == na_expected