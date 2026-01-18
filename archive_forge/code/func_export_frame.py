from typing import Dict
import numpy as np
import pandas
from modin.core.dataframe.pandas.interchange.dataframe_protocol.from_dataframe import (
from modin.tests.experimental.hdk_on_native.utils import ForceHdkImport
def export_frame(md_df, from_hdk=False, **kwargs):
    """
    Construct ``pandas.DataFrame`` from ``modin.pandas.DataFrame`` using DataFrame exchange protocol.

    Parameters
    ----------
    md_df : modin.pandas.DataFrame
        DataFrame to convert to pandas.
    from_hdk : bool, default: False
        Whether to forcibly use data exported from HDK. If `True`, import DataFrame's
        data into HDK and then export it back, so the origin for underlying `md_df`
        data is HDK.
    **kwargs : dict
        Additional parameters to pass to the ``from_dataframe_to_pandas`` function.

    Returns
    -------
    pandas.DataFrame
    """
    if not from_hdk:
        return from_dataframe_to_pandas_assert_chunking(md_df, **kwargs)
    with ForceHdkImport(md_df) as instance:
        md_df_exported = instance.export_frames()[0]
        exported_df = from_dataframe_to_pandas_assert_chunking(md_df_exported, **kwargs)
    return exported_df