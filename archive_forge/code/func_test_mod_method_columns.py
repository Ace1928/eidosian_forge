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
@pytest.mark.parametrize('fill_value', fill_values)
def test_mod_method_columns(self, fill_value):

    def mod1(df, fill_value, **kwargs):
        return df['a'].mod(df['b'], fill_value=fill_value)

    def mod2(df, fill_value, **kwargs):
        return df[['a', 'c']].mod(df[['b', 'a']], fill_value=fill_value)
    run_and_compare(mod1, data=self.data, fill_value=fill_value)
    run_and_compare(mod2, data=self.data, fill_value=fill_value)