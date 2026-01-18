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
@pytest.mark.parametrize('data_has_nulls', [True, False])
def test_export_unaligned_at_chunks(data_has_nulls):
    """
    Test export from DataFrame exchange protocol when internal PyArrow table's chunks are unaligned.

    Arrow table allows for its columns to be chunked independently. Unaligned chunking means that
    each column has its individual chunking and so some preprocessing is required in order
    to emulate equaly chunked columns in the protocol.
    """
    data = get_data_of_all_types(has_nulls=data_has_nulls, exclude_dtypes=['category'])
    pd_df = pandas.DataFrame(data)
    chunk_groups = [1, 2, 7]
    chunk_col_ilocs = [slice(i * len(pd_df.columns) // len(chunk_groups), (i + 1) * len(pd_df.columns) // len(chunk_groups)) for i in range(len(chunk_groups))]
    pd_chunk_groups = [split_df_into_chunks(pd_df.iloc[:, cols], n_chunks) for n_chunks, cols in zip(chunk_groups, chunk_col_ilocs)]
    at_chunk_groups = [pa.concat_tables([pa.Table.from_pandas(pd_df) for pd_df in chunk_group]) for chunk_group in pd_chunk_groups]
    chunked_at = at_chunk_groups[0]
    for _at in at_chunk_groups[1:]:
        for field in _at.schema:
            chunked_at = chunked_at.append_column(field, _at[field.name])
    md_df = from_arrow(chunked_at)
    internal_at = md_df._query_compiler._modin_frame._partitions[0][0].get()
    for n_chunks_group, cols in zip(chunk_groups, chunk_col_ilocs):
        for col in internal_at.select(range(cols.start, cols.stop)).columns:
            assert len(col.chunks) == n_chunks_group
    n_chunks = md_df.__dataframe__().num_chunks()
    exported_df = export_frame(md_df)
    df_equals(md_df, exported_df)
    exported_df = export_frame(md_df, n_chunks=n_chunks)
    df_equals(md_df, exported_df)
    exported_df = export_frame(md_df, n_chunks=n_chunks * 2)
    df_equals(md_df, exported_df)
    exported_df = export_frame(md_df, n_chunks=n_chunks * 3)
    df_equals(md_df, exported_df)