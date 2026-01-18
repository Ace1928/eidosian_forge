import pandas
import modin.pandas as pd
from modin.pandas.utils import from_dataframe
from modin.tests.pandas.utils import df_equals, test_data
from modin.tests.test_utils import warns_that_defaulting_to_pandas
def test_categorical_from_dataframe():
    modin_df = pd.DataFrame({'foo': pd.Series(['0', '1', '2', '3', '0', '3', '2', '3'], dtype='category')})
    eval_df_protocol(modin_df)