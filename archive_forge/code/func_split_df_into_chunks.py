from typing import Dict
import numpy as np
import pandas
from modin.core.dataframe.pandas.interchange.dataframe_protocol.from_dataframe import (
from modin.tests.experimental.hdk_on_native.utils import ForceHdkImport
def split_df_into_chunks(df, n_chunks):
    """
    Split passed DataFrame into `n_chunks` along row axis.

    Parameters
    ----------
    df : DataFrame
        DataFrame to split into chunks.
    n_chunks : int
        Number of chunks to split `df` into.

    Returns
    -------
    list of DataFrames
    """
    chunks = []
    for i in range(n_chunks):
        start = i * len(df) // n_chunks
        end = (i + 1) * len(df) // n_chunks
        chunks.append(df.iloc[start:end])
    return chunks