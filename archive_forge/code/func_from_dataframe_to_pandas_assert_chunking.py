from typing import Dict
import numpy as np
import pandas
from modin.core.dataframe.pandas.interchange.dataframe_protocol.from_dataframe import (
from modin.tests.experimental.hdk_on_native.utils import ForceHdkImport
def from_dataframe_to_pandas_assert_chunking(df, n_chunks=None, **kwargs):
    """
    Build a ``pandas.DataFrame`` from a `__dataframe__` object splitting it into `n_chunks`.

    The function asserts that the `df` was split exactly into `n_chunks` before converting them to pandas.

    Parameters
    ----------
    df : DataFrame
        Object supporting the exchange protocol, i.e. `__dataframe__` method.
    n_chunks : int, optional
        Number of chunks to split `df`.

    Returns
    -------
    pandas.DataFrame
    """
    if n_chunks is None:
        return from_dataframe_to_pandas(df, n_chunks=n_chunks, **kwargs)
    protocol_df = df.__dataframe__()
    chunks = list(protocol_df.get_chunks(n_chunks))
    assert len(chunks) == n_chunks
    pd_chunks = [None] * len(chunks)
    for i in range(len(chunks)):
        pd_chunks[i] = protocol_df_chunk_to_pandas(chunks[i], **kwargs)
    pd_df = pandas.concat(pd_chunks, axis=0, ignore_index=True)
    index_obj = protocol_df.metadata.get('modin.index', protocol_df.metadata.get('pandas.index', None))
    if index_obj is not None:
        pd_df.index = index_obj
    return pd_df