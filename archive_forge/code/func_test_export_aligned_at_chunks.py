import numpy as np
import pandas
import pyarrow as pa
import pytest
import modin.pandas as pd
from modin.core.dataframe.pandas.interchange.dataframe_protocol.from_dataframe import (
from modin.pandas.io import from_arrow, from_dataframe
from modin.tests.pandas.utils import df_equals
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from .utils import export_frame, get_data_of_all_types, split_df_into_chunks
@pytest.mark.parametrize('n_chunks', [2, 4, 7])
@pytest.mark.parametrize('data_has_nulls', [True, False])
def test_export_aligned_at_chunks(n_chunks, data_has_nulls):
    """Test export from DataFrame exchange protocol when internal PyArrow table is equaly chunked."""
    data = get_data_of_all_types(has_nulls=data_has_nulls, exclude_dtypes=['category'])
    pd_df = pandas.DataFrame(data)
    pd_chunks = split_df_into_chunks(pd_df, n_chunks)
    chunked_at = pa.concat_tables([pa.Table.from_pandas(pd_df) for pd_df in pd_chunks])
    md_df = from_arrow(chunked_at)
    assert len(md_df._query_compiler._modin_frame._partitions[0][0].get().column(0).chunks) == md_df.__dataframe__().num_chunks() == n_chunks
    exported_df = export_frame(md_df)
    df_equals(md_df, exported_df)
    exported_df = export_frame(md_df, n_chunks=n_chunks)
    df_equals(md_df, exported_df)
    exported_df = export_frame(md_df, n_chunks=n_chunks * 2)
    df_equals(md_df, exported_df)
    exported_df = export_frame(md_df, n_chunks=n_chunks * 3)
    df_equals(md_df, exported_df)