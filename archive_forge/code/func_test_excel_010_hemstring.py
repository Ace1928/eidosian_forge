from datetime import (
from functools import partial
from io import BytesIO
import os
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
from pandas.compat._constants import PY310
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._util import _writers
@pytest.mark.parametrize('use_headers', [True, False])
@pytest.mark.parametrize('r_idx_nlevels', [1, 2, 3])
@pytest.mark.parametrize('c_idx_nlevels', [1, 2, 3])
def test_excel_010_hemstring(self, merge_cells, c_idx_nlevels, r_idx_nlevels, use_headers, path):

    def roundtrip(data, header=True, parser_hdr=0, index=True):
        data.to_excel(path, header=header, merge_cells=merge_cells, index=index)
        with ExcelFile(path) as xf:
            return pd.read_excel(xf, sheet_name=xf.sheet_names[0], header=parser_hdr)
    parser_header = 0 if use_headers else None
    res = roundtrip(DataFrame([0]), use_headers, parser_header)
    assert res.shape == (1, 2)
    assert res.iloc[0, 0] is not np.nan
    nrows = 5
    ncols = 3
    if c_idx_nlevels == 1:
        columns = Index([f'a-{i}' for i in range(ncols)], dtype=object)
    else:
        columns = MultiIndex.from_arrays([range(ncols) for _ in range(c_idx_nlevels)], names=[f'i-{i}' for i in range(c_idx_nlevels)])
    if r_idx_nlevels == 1:
        index = Index([f'b-{i}' for i in range(nrows)], dtype=object)
    else:
        index = MultiIndex.from_arrays([range(nrows) for _ in range(r_idx_nlevels)], names=[f'j-{i}' for i in range(r_idx_nlevels)])
    df = DataFrame(np.ones((nrows, ncols)), columns=columns, index=index)
    if c_idx_nlevels > 1:
        msg = "Writing to Excel with MultiIndex columns and no index \\('index'=False\\) is not yet implemented."
        with pytest.raises(NotImplementedError, match=msg):
            roundtrip(df, use_headers, index=False)
    else:
        res = roundtrip(df, use_headers)
        if use_headers:
            assert res.shape == (nrows, ncols + r_idx_nlevels)
        else:
            assert res.shape == (nrows - 1, ncols + r_idx_nlevels)
        for r in range(len(res.index)):
            for c in range(len(res.columns)):
                assert res.iloc[r, c] is not np.nan