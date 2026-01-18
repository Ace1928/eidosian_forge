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
@pytest.mark.parametrize('cmp_fn', cmp_fn_values)
@pytest.mark.parametrize('value', [2, 2.2, 'a'])
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_cmp_mixed_types(self, cmp_fn, value, data):

    def cmp(df, cmp_fn, value, **kwargs):
        return getattr(df, cmp_fn)(value)
    run_and_compare(cmp, data=data, cmp_fn=cmp_fn, value=value)