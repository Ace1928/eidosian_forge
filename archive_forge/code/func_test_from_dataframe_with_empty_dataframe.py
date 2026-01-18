import pandas
import modin.pandas as pd
from modin.pandas.utils import from_dataframe
from modin.tests.pandas.utils import df_equals, test_data
from modin.tests.test_utils import warns_that_defaulting_to_pandas
def test_from_dataframe_with_empty_dataframe():
    modin_df = pd.DataFrame({'foo_col': pd.Series([], dtype='int64')})
    with warns_that_defaulting_to_pandas():
        eval_df_protocol(modin_df)