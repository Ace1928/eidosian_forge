import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from .utils import (
@pytest.mark.parametrize('method', ['corr', 'cov'])
def test_dataframe_corr_cov_with_self(method):
    mdf, pdf = create_test_dfs(test_data['float_nan_data'])
    with warns_that_defaulting_to_pandas():
        eval_general(mdf, pdf, lambda df, other: getattr(df.expanding(), method)(other=other), other=pdf, md_extra_kwargs={'other': mdf})