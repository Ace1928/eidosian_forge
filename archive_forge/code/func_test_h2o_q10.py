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
def test_h2o_q10(self):
    df = self._get_h2o_df()
    ref = df.groupby(['id1', 'id2', 'id3', 'id4', 'id5', 'id6'], observed=True).agg({'v3': 'sum', 'v1': 'count'})
    ref.reset_index(inplace=True)
    modin_df = pd.DataFrame(df)
    modin_df = modin_df.groupby(['id1', 'id2', 'id3', 'id4', 'id5', 'id6'], observed=True).agg({'v3': 'sum', 'v1': 'count'})
    modin_df.reset_index(inplace=True)
    exp = to_pandas(modin_df)
    exp['id1'] = exp['id1'].astype('category')
    exp['id2'] = exp['id2'].astype('category')
    exp['id3'] = exp['id3'].astype('category')
    df_equals(ref, exp)