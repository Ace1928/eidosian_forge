from __future__ import annotations
from typing import TYPE_CHECKING, Any
from streamlit import type_util
from streamlit.elements.lib import pandas_styler_utils
from streamlit.proto.Components_pb2 import ArrowTable as ArrowTableProto
def arrow_proto_to_dataframe(proto: ArrowTableProto) -> DataFrame:
    """Convert ArrowTable proto to pandas.DataFrame.

    Parameters
    ----------
    proto : proto.ArrowTable
        Output. pandas.DataFrame

    """
    if type_util.is_pyarrow_version_less_than('14.0.1'):
        raise RuntimeError('The installed pyarrow version is not compatible with this component. Please upgrade to 14.0.1 or higher: pip install -U pyarrow')
    import pandas as pd
    data = type_util.bytes_to_data_frame(proto.data)
    index = type_util.bytes_to_data_frame(proto.index)
    columns = type_util.bytes_to_data_frame(proto.columns)
    return pd.DataFrame(data.values, index=index.values.T.tolist(), columns=columns.values.T.tolist())