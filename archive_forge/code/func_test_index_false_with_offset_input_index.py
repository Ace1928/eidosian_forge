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
@pytest.mark.parametrize('offset_index', [list(range(10, 13)), [str(i) for i in range(10, 13)]])
def test_index_false_with_offset_input_index(parser, offset_index, geom_df):
    """
    Tests that the output does not contain the `<index>` field when the index of the
    input Dataframe has an offset.

    This is a regression test for issue #42458.
    """
    expected = "<?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <shape>square</shape>\n    <degrees>360</degrees>\n    <sides>4.0</sides>\n  </row>\n  <row>\n    <shape>circle</shape>\n    <degrees>360</degrees>\n    <sides/>\n  </row>\n  <row>\n    <shape>triangle</shape>\n    <degrees>180</degrees>\n    <sides>3.0</sides>\n  </row>\n</data>"
    offset_geom_df = geom_df.copy()
    offset_geom_df.index = Index(offset_index)
    output = offset_geom_df.to_xml(index=False, parser=parser)
    output = equalize_decl(output)
    assert output == expected