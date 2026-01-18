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
@pytest.mark.parametrize('force_hdk', [False, True])
def test_arithmetic_ops(self, force_hdk):

    def compute(df, operation, **kwargs):
        df = getattr(df, operation)(3)
        return df
    for op in ('__add__', '__sub__', '__mul__', '__pow__', '__truediv__', '__floordiv__'):
        run_and_compare(compute, {'A': [1, 2, 3, 4, 5]}, operation=op, force_hdk_execute=force_hdk)