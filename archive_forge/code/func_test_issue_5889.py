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
@pytest.mark.parametrize('transform', [True, False])
@pytest.mark.parametrize('sort_last', [True, False])
def test_issue_5889(self, transform, sort_last):
    with ensure_clean('.csv') as file:
        data = {'a': [1, 2, 3], 'b': [1, 2, 3]} if transform else {'a': [1, 2, 3]}
        pandas.DataFrame(data).to_csv(file, index=False)

        def test_concat(lib, **kwargs):
            if transform:

                def read_csv():
                    return lib.read_csv(file)['b']
            else:

                def read_csv():
                    return lib.read_csv(file)
            df = read_csv()
            for _ in range(100):
                df = lib.concat([df, read_csv()])
            if sort_last:
                df = lib.concat([df, read_csv()], sort=True)
            return df
        run_and_compare(test_concat, data={})