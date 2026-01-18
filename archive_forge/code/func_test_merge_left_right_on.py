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
@pytest.mark.parametrize('how', how_values)
@pytest.mark.parametrize('left_on, right_on', [['a', 'c'], [['a', 'b'], ['c', 'b']]])
def test_merge_left_right_on(self, how, left_on, right_on):

    def merge(df1, df2, how, left_on, right_on, **kwargs):
        return df1.merge(df2, how=how, left_on=left_on, right_on=right_on)
    run_and_compare(merge, data=self.left_data, data2=self.right_data, how=how, left_on=left_on, right_on=right_on)
    run_and_compare(merge, data=self.right_data, data2=self.left_data, how=how, left_on=right_on, right_on=left_on)