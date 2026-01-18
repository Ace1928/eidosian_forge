import json
import os
import random
import hashlib
import warnings
from typing import Union, MutableMapping, Optional, Dict, Sequence, TYPE_CHECKING, List
import pandas as pd
from toolz import curried
from typing import TypeVar
from ._importers import import_pyarrow_interchange
from .core import sanitize_dataframe, sanitize_arrow_table, DataFrameLike
from .core import sanitize_geo_interface
from .deprecation import AltairDeprecationWarning
from .plugin_registry import PluginRegistry
from typing import Protocol, TypedDict, Literal
@curried.curry
def limit_rows(data: TDataType, max_rows: Optional[int]=5000) -> TDataType:
    """Raise MaxRowsError if the data model has more than max_rows.

    If max_rows is None, then do not perform any check.
    """
    check_data_type(data)

    def raise_max_rows_error():
        raise MaxRowsError(f'The number of rows in your dataset is greater than the maximum allowed ({max_rows}).\n\nTry enabling the VegaFusion data transformer which raises this limit by pre-evaluating data\ntransformations in Python.\n    >> import altair as alt\n    >> alt.data_transformers.enable("vegafusion")\n\nOr, see https://altair-viz.github.io/user_guide/large_datasets.html for additional information\non how to plot large datasets.')
    if hasattr(data, '__geo_interface__'):
        if data.__geo_interface__['type'] == 'FeatureCollection':
            values = data.__geo_interface__['features']
        else:
            values = data.__geo_interface__
    elif isinstance(data, pd.DataFrame):
        values = data
    elif isinstance(data, dict):
        if 'values' in data:
            values = data['values']
        else:
            return data
    elif isinstance(data, DataFrameLike):
        pa_table = arrow_table_from_dfi_dataframe(data)
        if max_rows is not None and pa_table.num_rows > max_rows:
            raise_max_rows_error()
        return pa_table
    if max_rows is not None and len(values) > max_rows:
        raise_max_rows_error()
    return data