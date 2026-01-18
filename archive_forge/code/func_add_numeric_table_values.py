import collections
import datetime
import enum
import itertools
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Text, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import (
from ...utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available, logging
def add_numeric_table_values(table, min_consolidation_fraction=0.7, debug_info=None):
    """
    Parses text in table column-wise and adds the consolidated values. Consolidation refers to finding values with a
    common types (date or number)

    Args:
        table:
            Table to annotate.
        min_consolidation_fraction:
            Fraction of cells in a column that need to have consolidated value.
        debug_info:
            Additional information used for logging.
    """
    table = table.copy()
    filter_invalid_unicode_from_table(table)
    for row_index, row in table.iterrows():
        for col_index, cell in enumerate(row):
            table.iloc[row_index, col_index] = Cell(text=cell)
    for col_index, column in enumerate(table.columns):
        column_values = _consolidate_numeric_values(_get_column_values(table, col_index), min_consolidation_fraction=min_consolidation_fraction, debug_info=(debug_info, column))
        for row_index, numeric_value in column_values.items():
            table.iloc[row_index, col_index].numeric_value = numeric_value
    return table