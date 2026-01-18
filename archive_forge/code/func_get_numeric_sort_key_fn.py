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
def get_numeric_sort_key_fn(numeric_values):
    """
    Creates a function that can be used as a sort key or to compare the values. Maps to primitive types and finds the
    biggest common subset. Consider the values "05/05/2010" and "August 2007". With the corresponding primitive values
    (2010.,5.,5.) and (2007.,8., None). These values can be compared by year and date so we map to the sequence (2010.,
    5.), (2007., 8.). If we added a third value "2006" with primitive value (2006., None, None), we could only compare
    by the year so we would map to (2010.,), (2007.,) and (2006.,).

    Args:
     numeric_values: Values to compare

    Returns:
     A function that can be used as a sort key function (mapping numeric values to a comparable tuple)

    Raises:
      ValueError if values don't have a common type or are not comparable.
    """
    value_types = _get_all_types(numeric_values)
    if len(value_types) != 1:
        raise ValueError(f'No common value type in {numeric_values}')
    value_type = next(iter(value_types))
    if value_type == NUMBER_TYPE:
        return _get_value_as_primitive_value
    valid_indexes = set(range(_DATE_TUPLE_SIZE))
    for numeric_value in numeric_values:
        value = _get_value_as_primitive_value(numeric_value)
        assert isinstance(value, tuple)
        for tuple_index, inner_value in enumerate(value):
            if inner_value is None:
                valid_indexes.discard(tuple_index)
    if not valid_indexes:
        raise ValueError(f'No common value in {numeric_values}')

    def _sort_key_fn(numeric_value):
        value = _get_value_as_primitive_value(numeric_value)
        return tuple((value[index] for index in valid_indexes))
    return _sort_key_fn