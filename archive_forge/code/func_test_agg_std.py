import os
import re
import numpy as np
import pandas
import pyarrow
import pytest
from pandas._testing import ensure_clean
from pandas.core.dtypes.common import is_list_like
from pyhdk import __version__ as hdk_version
from modin.config import StorageFormat
from modin.tests.interchange.dataframe_protocol.hdk.utils import split_df_into_chunks
from modin.tests.pandas.utils import (
from .utils import ForceHdkImport, eval_io, run_and_compare, set_execution_mode
import modin.pandas as pd
from modin.experimental.core.execution.native.implementations.hdk_on_native.calcite_serializer import (
from modin.experimental.core.execution.native.implementations.hdk_on_native.df_algebra import (
from modin.experimental.core.execution.native.implementations.hdk_on_native.partitioning.partition_manager import (
from modin.pandas.io import from_arrow
from modin.tests.pandas.utils import (
from modin.utils import try_cast_to_pandas
def test_agg_std(self):

    def std(df, **kwargs):
        df = df.groupby('a').agg({'b': 'std', 'c': 'std'})
        if not isinstance(df, pandas.DataFrame):
            df = to_pandas(df)
        df['b'] = df['b'].apply(lambda x: round(x, 10))
        df['c'] = df['c'].apply(lambda x: round(x, 10))
        return df
    run_and_compare(std, data=self.std_data, force_lazy=False)