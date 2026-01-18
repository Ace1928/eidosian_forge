from toolz import curried
import uuid
from weakref import WeakValueDictionary
from typing import (
from altair.utils._importers import import_vegafusion
from altair.utils.core import DataFrameLike
from altair.utils.data import DataType, ToValuesReturnType, MaxRowsError
from altair.vegalite.data import default_data_transformer
def handle_row_limit_exceeded(row_limit: int, warnings: list):
    for warning in warnings:
        if warning.get('type') == 'RowLimitExceeded':
            raise MaxRowsError(f'The number of dataset rows after filtering and aggregation exceeds\nthe current limit of {row_limit}. Try adding an aggregation to reduce\nthe size of the dataset that must be loaded into the browser. Or, disable\nthe limit by calling alt.data_transformers.disable_max_rows(). Note that\ndisabling this limit may cause the browser to freeze or crash.')