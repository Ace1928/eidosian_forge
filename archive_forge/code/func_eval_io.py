import datetime
import pandas
import pytest
from pandas.core.dtypes.common import is_datetime64_any_dtype, is_object_dtype
import modin.pandas as pd
from modin.tests.pandas.utils import df_equals
from modin.tests.pandas.utils import eval_io as general_eval_io
from modin.utils import try_cast_to_pandas
def eval_io(fn_name, comparator=df_equals, cast_to_str=False, expected_exception=None, check_kwargs_callable=True, modin_warning=None, md_extra_kwargs=None, *args, **kwargs):
    """
    Evaluate I/O operation and do equality check after importing Modin's data to HDK.

    Notes
    -----
    For parameters description please refer to ``modin.tests.pandas.utils.eval_io``.
    """

    def hdk_comparator(df1, df2, **kwargs):
        """Evaluate equality comparison of the passed frames after importing the Modin's one to HDK."""
        with ForceHdkImport(df1, df2):
            dfs = align_datetime_dtypes(df1, df2)
            cols = {c for df in dfs for c, t in df.dtypes.items() if is_object_dtype(t)}
            if len(cols) != 0:
                cols = pandas.Index(cols)
                for df in dfs:
                    df[cols] = df[cols].fillna('')
            cols = {c for df in dfs for c, t in df.dtypes.items() if isinstance(t, pandas.CategoricalDtype)}
            if len(cols) != 0:
                cols = pandas.Index(cols)
                for df in dfs:
                    df[cols] = df[cols].astype(str)
                comparator(*dfs, **kwargs)
    general_eval_io(fn_name, *args, comparator=hdk_comparator, cast_to_str=cast_to_str, expected_exception=expected_exception, check_kwargs_callable=check_kwargs_callable, modin_warning=modin_warning, md_extra_kwargs=md_extra_kwargs, **kwargs)