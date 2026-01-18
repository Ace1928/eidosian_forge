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
def test_dropna_multiindex(self):
    index = generate_multiindex(len(self.data['col1']))
    md_df = pd.DataFrame(self.data, index=index)
    pd_df = pandas.DataFrame(self.data, index=index)
    md_res = md_df.dropna()._to_pandas()
    pd_res = pd_df.dropna()
    md_res.index = pandas.MultiIndex.from_tuples(md_res.index.values, names=md_res.index.names)
    df_equals(md_res, pd_res)