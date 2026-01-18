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
def test_elems_cols_nan_output(parser, geom_df):
    elems_cols_expected = "<?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <degrees>360</degrees>\n    <sides>4.0</sides>\n    <shape>square</shape>\n  </row>\n  <row>\n    <degrees>360</degrees>\n    <sides/>\n    <shape>circle</shape>\n  </row>\n  <row>\n    <degrees>180</degrees>\n    <sides>3.0</sides>\n    <shape>triangle</shape>\n  </row>\n</data>"
    output = geom_df.to_xml(index=False, elem_cols=['degrees', 'sides', 'shape'], parser=parser)
    output = equalize_decl(output)
    assert output == elems_cols_expected