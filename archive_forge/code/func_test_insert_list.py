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
@pytest.mark.parametrize('data', [None, {'A': range(10)}, pandas.DataFrame({'A': range(10)})])
@pytest.mark.parametrize('index', [None, pandas.RangeIndex(10), pandas.RangeIndex(start=10, stop=0, step=-1)])
@pytest.mark.parametrize('value', [list(range(10)), pandas.Series(range(10))])
@pytest.mark.parametrize('part_type', [None, 'arrow', 'hdk'])
@pytest.mark.parametrize('insert_scalar', [True, False])
def test_insert_list(self, data, index, value, part_type, insert_scalar):

    def create():
        mdf, pdf = create_test_dfs(data, index=index)
        if part_type == 'arrow':
            mdf._query_compiler._modin_frame._partitions[0][0].get(True)
        elif part_type == 'hdk':
            mdf._query_compiler._modin_frame.force_import()
        return (mdf, pdf)

    def insert(loc, name, value):
        nonlocal mdf, pdf
        mdf.insert(loc, name, value)
        pdf.insert(loc, name, value)
        if insert_scalar:
            mdf[f'S{loc}'] = 1
            pdf[f'S{loc}'] = 1
    niter = 3
    mdf, pdf = create()
    for i in range(niter):
        insert(len(pdf.columns), f'B{i}', value)
    df_equals(mdf, pdf)
    mdf, pdf = create()
    for i in range(niter):
        insert(0, f'C{i}', value)
    df_equals(mdf, pdf)
    mdf, pdf = create()
    for i in range(niter):
        insert(len(pdf.columns), f'B{i}', value)
        insert(0, f'C{i}', value)
        insert(len(pdf.columns) // 2, f'D{i}', value)
    df_equals(mdf, pdf)