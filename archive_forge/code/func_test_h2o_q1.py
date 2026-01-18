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
def test_h2o_q1(self):
    lhs = self._get_h2o_df(self.h2o_data)
    rhs = self._get_h2o_df(self.h2o_data_small)
    ref = lhs.merge(rhs, on='id1')
    self._fix_category_cols(ref)
    modin_lhs = pd.DataFrame(lhs)
    modin_rhs = pd.DataFrame(rhs)
    modin_res = modin_lhs.merge(modin_rhs, on='id1')
    exp = to_pandas(modin_res)
    self._fix_category_cols(exp)
    df_equals(ref, exp)