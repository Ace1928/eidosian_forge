import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
def test_ix_categorical_index_non_unique(self, infer_string):
    with option_context('future.infer_string', infer_string):
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), index=list('ABA'), columns=list('XYX'))
        cdf = df.copy()
        cdf.index = CategoricalIndex(df.index)
        cdf.columns = CategoricalIndex(df.columns)
        exp_index = CategoricalIndex(list('AA'), categories=['A', 'B'])
        expect = DataFrame(df.loc['A', :], columns=cdf.columns, index=exp_index)
        tm.assert_frame_equal(cdf.loc['A', :], expect)
        exp_columns = CategoricalIndex(list('XX'), categories=['X', 'Y'])
        expect = DataFrame(df.loc[:, 'X'], index=cdf.index, columns=exp_columns)
        tm.assert_frame_equal(cdf.loc[:, 'X'], expect)
        expect = DataFrame(df.loc[['A', 'B'], :], columns=cdf.columns, index=CategoricalIndex(list('AAB')))
        tm.assert_frame_equal(cdf.loc[['A', 'B'], :], expect)
        expect = DataFrame(df.loc[:, ['X', 'Y']], index=cdf.index, columns=CategoricalIndex(list('XXY')))
        tm.assert_frame_equal(cdf.loc[:, ['X', 'Y']], expect)