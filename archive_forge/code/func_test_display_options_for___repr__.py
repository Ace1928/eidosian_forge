import io
import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions
from modin.pandas.utils import SET_DATAFRAME_ATTRIBUTE_WARNING
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
@pytest.mark.parametrize('expand_frame_repr', [False, True])
@pytest.mark.parametrize('max_rows_columns', [(5, 5), (10, 10), (50, 50), (51, 51), (52, 52), (75, 75), (None, None)])
@pytest.mark.parametrize('frame_size', [101, 102])
def test_display_options_for___repr__(max_rows_columns, expand_frame_repr, frame_size):
    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(frame_size, frame_size))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    context_arg = ['display.max_rows', max_rows_columns[0], 'display.max_columns', max_rows_columns[1], 'display.expand_frame_repr', expand_frame_repr]
    with pd.option_context(*context_arg):
        modin_df_repr = repr(modin_df)
    with pandas.option_context(*context_arg):
        pandas_df_repr = repr(pandas_df)
    assert modin_df_repr == pandas_df_repr