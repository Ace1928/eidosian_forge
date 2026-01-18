from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, TypeVar
from streamlit import type_util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
def marshall_styler(proto: ArrowProto, styler: Styler, default_uuid: str) -> None:
    """Marshall pandas.Styler into an Arrow proto.

    Parameters
    ----------
    proto : proto.Arrow
        Output. The protobuf for Streamlit Arrow proto.

    styler : pandas.Styler
        Helps style a DataFrame or Series according to the data with HTML and CSS.

    default_uuid : str
        If pandas.Styler uuid is not provided, this value will be used.

    """
    import pandas as pd
    styler_data_df: pd.DataFrame = styler.data
    if styler_data_df.size > int(pd.options.styler.render.max_elements):
        raise StreamlitAPIException(f'The dataframe has `{styler_data_df.size}` cells, but the maximum number of cells allowed to be rendered by Pandas Styler is configured to `{pd.options.styler.render.max_elements}`. To allow more cells to be styled, you can change the `"styler.render.max_elements"` config. For example: `pd.set_option("styler.render.max_elements", {styler_data_df.size})`')
    _marshall_uuid(proto, styler, default_uuid)
    styler._compute()
    pandas_styles = styler._translate(False, False)
    _marshall_caption(proto, styler)
    _marshall_styles(proto, styler, pandas_styles)
    _marshall_display_values(proto, styler_data_df, pandas_styles)