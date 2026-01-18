from __future__ import annotations
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import (
from typing_extensions import TypeAlias
from streamlit import logger as _logger
from streamlit import type_util
from streamlit.deprecation_util import deprecate_func_name
from streamlit.elements.form import current_form_id
from streamlit.elements.lib.column_config_utils import (
from streamlit.elements.lib.pandas_styler_utils import marshall_styler
from streamlit.elements.utils import check_callback_rules, check_session_state_rules
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import DataFormat, DataFrameGenericAlias, Key, is_type, to_key
from streamlit.util import calc_md5
@gather_metrics('data_editor')
def data_editor(self, data: DataTypes, *, width: int | None=None, height: int | None=None, use_container_width: bool=False, hide_index: bool | None=None, column_order: Iterable[str] | None=None, column_config: ColumnConfigMappingInput | None=None, num_rows: Literal['fixed', 'dynamic']='fixed', disabled: bool | Iterable[str]=False, key: Key | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None) -> DataTypes:
    """Display a data editor widget.

        The data editor widget allows you to edit dataframes and many other data structures in a table-like UI.

        .. warning::
            When going from ``st.experimental_data_editor`` to ``st.data_editor`` in
            1.23.0, the data editor's representation in ``st.session_state`` was changed.
            The ``edited_cells`` dictionary is now called ``edited_rows`` and uses a
            different format (``{0: {"column name": "edited value"}}`` instead of
            ``{"0:1": "edited value"}``). You may need to adjust the code if your app uses
            ``st.experimental_data_editor`` in combination with ``st.session_state``.

        Parameters
        ----------
        data : pandas.DataFrame, pandas.Series, pandas.Styler, pandas.Index, pyarrow.Table, numpy.ndarray, pyspark.sql.DataFrame, snowflake.snowpark.DataFrame, list, set, tuple, dict, or None
            The data to edit in the data editor.

            .. note::
                - Styles from ``pandas.Styler`` will only be applied to non-editable columns.
                - Mixing data types within a column can make the column uneditable.
                - Additionally, the following data types are not yet supported for editing:
                  complex, list, tuple, bytes, bytearray, memoryview, dict, set, frozenset,
                  fractions.Fraction, pandas.Interval, and pandas.Period.
                - To prevent overflow in JavaScript, columns containing datetime.timedelta
                  and pandas.Timedelta values will default to uneditable but this can be
                  changed through column configuration.

        width : int or None
            Desired width of the data editor expressed in pixels. If None, the width will
            be automatically determined.

        height : int or None
            Desired height of the data editor expressed in pixels. If None, the height will
            be automatically determined.

        use_container_width : bool
            If True, set the data editor width to the width of the parent container.
            This takes precedence over the width argument. Defaults to False.

        hide_index : bool or None
            Whether to hide the index column(s). If None (default), the visibility of
            index columns is automatically determined based on the data.

        column_order : Iterable of str or None
            Specifies the display order of columns. This also affects which columns are
            visible. For example, ``column_order=("col2", "col1")`` will display 'col2'
            first, followed by 'col1', and will hide all other non-index columns. If
            None (default), the order is inherited from the original data structure.

        column_config : dict or None
            Configures how columns are displayed, e.g. their title, visibility, type, or
            format, as well as editing properties such as min/max value or step.
            This needs to be a dictionary where each key is a column name and the value
            is one of:

            * ``None`` to hide the column.

            * A string to set the display label of the column.

            * One of the column types defined under ``st.column_config``, e.g.
              ``st.column_config.NumberColumn("Dollar values”, format=”$ %d")`` to show
              a column as dollar amounts. See more info on the available column types
              and config options `here <https://docs.streamlit.io/library/api-reference/data/st.column_config>`_.

            To configure the index column(s), use ``_index`` as the column name.

        num_rows : "fixed" or "dynamic"
            Specifies if the user can add and delete rows in the data editor.
            If "fixed", the user cannot add or delete rows. If "dynamic", the user can
            add and delete rows in the data editor, but column sorting is disabled.
            Defaults to "fixed".

        disabled : bool or Iterable of str
            Controls the editing of columns. If True, editing is disabled for all columns.
            If an Iterable of column names is provided (e.g., ``disabled=("col1", "col2"))``,
            only the specified columns will be disabled for editing. If False (default),
            all columns that support editing are editable.

        key : str
            An optional string to use as the unique key for this widget. If this
            is omitted, a key will be generated for the widget based on its
            content. Multiple widgets of the same type may not share the same
            key.

        on_change : callable
            An optional callback invoked when this data_editor's value changes.

        args : tuple
            An optional tuple of args to pass to the callback.

        kwargs : dict
            An optional dict of kwargs to pass to the callback.

        Returns
        -------
        pandas.DataFrame, pandas.Series, pyarrow.Table, numpy.ndarray, list, set, tuple, or dict.
            The edited data. The edited data is returned in its original data type if
            it corresponds to any of the supported return types. All other data types
            are returned as a ``pandas.DataFrame``.

        Examples
        --------
        >>> import streamlit as st
        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame(
        >>>     [
        >>>        {"command": "st.selectbox", "rating": 4, "is_widget": True},
        >>>        {"command": "st.balloons", "rating": 5, "is_widget": False},
        >>>        {"command": "st.time_input", "rating": 3, "is_widget": True},
        >>>    ]
        >>> )
        >>> edited_df = st.data_editor(df)
        >>>
        >>> favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]
        >>> st.markdown(f"Your favorite command is **{favorite_command}** 🎈")

        .. output::
           https://doc-data-editor.streamlit.app/
           height: 350px

        You can also allow the user to add and delete rows by setting ``num_rows`` to "dynamic":

        >>> import streamlit as st
        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame(
        >>>     [
        >>>        {"command": "st.selectbox", "rating": 4, "is_widget": True},
        >>>        {"command": "st.balloons", "rating": 5, "is_widget": False},
        >>>        {"command": "st.time_input", "rating": 3, "is_widget": True},
        >>>    ]
        >>> )
        >>> edited_df = st.data_editor(df, num_rows="dynamic")
        >>>
        >>> favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]
        >>> st.markdown(f"Your favorite command is **{favorite_command}** 🎈")

        .. output::
           https://doc-data-editor1.streamlit.app/
           height: 450px

        Or you can customize the data editor via ``column_config``, ``hide_index``, ``column_order``, or ``disabled``:

        >>> import pandas as pd
        >>> import streamlit as st
        >>>
        >>> df = pd.DataFrame(
        >>>     [
        >>>         {"command": "st.selectbox", "rating": 4, "is_widget": True},
        >>>         {"command": "st.balloons", "rating": 5, "is_widget": False},
        >>>         {"command": "st.time_input", "rating": 3, "is_widget": True},
        >>>     ]
        >>> )
        >>> edited_df = st.data_editor(
        >>>     df,
        >>>     column_config={
        >>>         "command": "Streamlit Command",
        >>>         "rating": st.column_config.NumberColumn(
        >>>             "Your rating",
        >>>             help="How much do you like this command (1-5)?",
        >>>             min_value=1,
        >>>             max_value=5,
        >>>             step=1,
        >>>             format="%d ⭐",
        >>>         ),
        >>>         "is_widget": "Widget ?",
        >>>     },
        >>>     disabled=["command", "is_widget"],
        >>>     hide_index=True,
        >>> )
        >>>
        >>> favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]
        >>> st.markdown(f"Your favorite command is **{favorite_command}** 🎈")


        .. output::
           https://doc-data-editor-config.streamlit.app/
           height: 350px

        """
    import pandas as pd
    import pyarrow as pa
    key = to_key(key)
    check_callback_rules(self.dg, on_change)
    check_session_state_rules(default_value=None, key=key, writes_allowed=False)
    if column_order is not None:
        column_order = list(column_order)
    column_config_mapping: ColumnConfigMapping = {}
    data_format = type_util.determine_data_format(data)
    if data_format == DataFormat.UNKNOWN:
        raise StreamlitAPIException(f'The data type ({type(data).__name__}) or format is not supported by the data editor. Please convert your data into a Pandas Dataframe or another supported data format.')
    data_df = type_util.convert_anything_to_df(data, ensure_copy=True)
    if not _is_supported_index(data_df.index):
        raise StreamlitAPIException(f'The type of the dataframe index - {type(data_df.index).__name__} - is not yet supported by the data editor.')
    _check_column_names(data_df)
    column_config_mapping = process_config_mapping(column_config)
    apply_data_specific_configs(column_config_mapping, data_df, data_format, check_arrow_compatibility=True)
    _fix_column_headers(data_df)
    if isinstance(data_df.index, pd.RangeIndex) and num_rows == 'dynamic':
        update_column_config(column_config_mapping, INDEX_IDENTIFIER, {'hidden': True})
    if hide_index is not None:
        update_column_config(column_config_mapping, INDEX_IDENTIFIER, {'hidden': hide_index})
    if not isinstance(disabled, bool):
        for column in disabled:
            update_column_config(column_config_mapping, column, {'disabled': True})
    arrow_table = pa.Table.from_pandas(data_df)
    dataframe_schema = determine_dataframe_schema(data_df, arrow_table.schema)
    _check_type_compatibilities(data_df, column_config_mapping, dataframe_schema)
    arrow_bytes = type_util.pyarrow_table_to_bytes(arrow_table)
    ctx = get_script_run_ctx()
    id = compute_widget_id('data_editor', user_key=key, data=arrow_bytes, width=width, height=height, use_container_width=use_container_width, column_order=column_order, column_config_mapping=str(column_config_mapping), num_rows=num_rows, key=key, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
    proto = ArrowProto()
    proto.id = id
    proto.use_container_width = use_container_width
    if width:
        proto.width = width
    if height:
        proto.height = height
    if column_order:
        proto.column_order[:] = column_order
    proto.disabled = disabled is True
    proto.editing_mode = ArrowProto.EditingMode.DYNAMIC if num_rows == 'dynamic' else ArrowProto.EditingMode.FIXED
    proto.form_id = current_form_id(self.dg)
    if type_util.is_pandas_styler(data):
        styler_uuid = calc_md5(key or self.dg._get_delta_path_str())[:10]
        data.set_uuid(styler_uuid)
        marshall_styler(proto, data, styler_uuid)
    proto.data = arrow_bytes
    marshall_column_config(proto, column_config_mapping)
    serde = DataEditorSerde()
    widget_state = register_widget('data_editor', proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
    _apply_dataframe_edits(data_df, widget_state.value, dataframe_schema)
    self.dg._enqueue('arrow_data_frame', proto)
    return type_util.convert_df_to_data_format(data_df, data_format)