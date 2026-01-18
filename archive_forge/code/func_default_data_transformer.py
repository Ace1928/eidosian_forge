from toolz import curried
from ..utils.core import sanitize_dataframe
from ..utils.data import (
from ..utils.data import DataTransformerRegistry as _DataTransformerRegistry
from ..utils.data import DataType, ToValuesReturnType
from ..utils.plugin_registry import PluginEnabler
@curried.curry
def default_data_transformer(data: DataType, max_rows: int=5000) -> ToValuesReturnType:
    return curried.pipe(data, limit_rows(max_rows=max_rows), to_values)