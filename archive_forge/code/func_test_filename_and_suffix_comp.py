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
def test_filename_and_suffix_comp(parser, compression_only, geom_df, compression_to_extension):
    compfile = 'xml.' + compression_to_extension[compression_only]
    with tm.ensure_clean(filename=compfile) as path:
        geom_df.to_xml(path, parser=parser, compression=compression_only)
        with get_handle(path, 'r', compression=compression_only) as handle_obj:
            output = handle_obj.handle.read()
    output = equalize_decl(output)
    assert geom_xml == output.strip()