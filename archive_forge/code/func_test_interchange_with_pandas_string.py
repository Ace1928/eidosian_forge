import pandas
import modin.pandas as pd
from modin.pandas.utils import from_dataframe
from modin.tests.pandas.utils import df_equals, test_data
from modin.tests.test_utils import warns_that_defaulting_to_pandas
def test_interchange_with_pandas_string():
    modin_df = pd.DataFrame({'fips': ['01001']})
    pandas_df = pandas.api.interchange.from_dataframe(modin_df.__dataframe__())
    df_equals(modin_df, pandas_df)