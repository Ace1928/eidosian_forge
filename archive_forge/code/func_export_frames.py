import datetime
import pandas
import pytest
from pandas.core.dtypes.common import is_datetime64_any_dtype, is_object_dtype
import modin.pandas as pd
from modin.tests.pandas.utils import df_equals
from modin.tests.pandas.utils import eval_io as general_eval_io
from modin.utils import try_cast_to_pandas
def export_frames(self):
    """
        Export tables from HDK that was imported by this instance.

        Returns
        -------
        list
            A list of Modin DataFrames whose payload is ``pyarrow.Table``
            that was just exported from HDK.
        """
    result = []
    for df in self._imported_frames:
        df = df[df.columns.tolist()]
        modin_frame = df._query_compiler._modin_frame
        mode = modin_frame._force_execution_mode
        modin_frame._force_execution_mode = 'hdk'
        modin_frame._execute()
        modin_frame._force_execution_mode = mode
        result.append(df)
    return result