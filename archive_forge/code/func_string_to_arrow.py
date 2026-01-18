import copy
import json
import re
import sys
from collections.abc import Iterable, Mapping
from collections.abc import Sequence as SequenceABC
from dataclasses import InitVar, dataclass, field, fields
from functools import reduce, wraps
from operator import mul
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
from typing import Sequence as Sequence_
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
import pyarrow_hotfix  # noqa: F401  # to fix vulnerability on pyarrow<14.0.1
from pandas.api.extensions import ExtensionArray as PandasExtensionArray
from pandas.api.extensions import ExtensionDtype as PandasExtensionDtype
from .. import config
from ..naming import camelcase_to_snakecase, snakecase_to_camelcase
from ..table import array_cast
from ..utils import logging
from ..utils.py_utils import asdict, first_non_null_value, zip_dict
from .audio import Audio
from .image import Image, encode_pil_image
from .translation import Translation, TranslationVariableLanguages
def string_to_arrow(datasets_dtype: str) -> pa.DataType:
    """
    string_to_arrow takes a datasets string dtype and converts it to a pyarrow.DataType.

    In effect, `dt == string_to_arrow(_arrow_to_datasets_dtype(dt))`

    This is necessary because the datasets.Value() primitive type is constructed using a string dtype

    Value(dtype=str)

    But Features.type (via `get_nested_type()` expects to resolve Features into a pyarrow Schema,
        which means that each Value() must be able to resolve into a corresponding pyarrow.DataType, which is the
        purpose of this function.
    """

    def _dtype_error_msg(dtype, pa_dtype, examples=None, urls=None):
        msg = f'{dtype} is not a validly formatted string representation of the pyarrow {pa_dtype} type.'
        if examples:
            examples = ', '.join(examples[:-1]) + ' or ' + examples[-1] if len(examples) > 1 else examples[0]
            msg += f'\nValid examples include: {examples}.'
        if urls:
            urls = ', '.join(urls[:-1]) + ' and ' + urls[-1] if len(urls) > 1 else urls[0]
            msg += f'\nFor more insformation, see: {urls}.'
        return msg
    if datasets_dtype in pa.__dict__:
        return pa.__dict__[datasets_dtype]()
    if datasets_dtype + '_' in pa.__dict__:
        return pa.__dict__[datasets_dtype + '_']()
    timestamp_matches = re.search('^timestamp\\[(.*)\\]$', datasets_dtype)
    if timestamp_matches:
        timestamp_internals = timestamp_matches.group(1)
        internals_matches = re.search('^(s|ms|us|ns),\\s*tz=([a-zA-Z0-9/_+\\-:]*)$', timestamp_internals)
        if timestamp_internals in ['s', 'ms', 'us', 'ns']:
            return pa.timestamp(timestamp_internals)
        elif internals_matches:
            return pa.timestamp(internals_matches.group(1), internals_matches.group(2))
        else:
            raise ValueError(_dtype_error_msg(datasets_dtype, 'timestamp', examples=['timestamp[us]', 'timestamp[us, tz=America/New_York'], urls=['https://arrow.apache.org/docs/python/generated/pyarrow.timestamp.html']))
    duration_matches = re.search('^duration\\[(.*)\\]$', datasets_dtype)
    if duration_matches:
        duration_internals = duration_matches.group(1)
        if duration_internals in ['s', 'ms', 'us', 'ns']:
            return pa.duration(duration_internals)
        else:
            raise ValueError(_dtype_error_msg(datasets_dtype, 'duration', examples=['duration[s]', 'duration[us]'], urls=['https://arrow.apache.org/docs/python/generated/pyarrow.duration.html']))
    time_matches = re.search('^time(.*)\\[(.*)\\]$', datasets_dtype)
    if time_matches:
        time_internals_bits = time_matches.group(1)
        if time_internals_bits == '32':
            time_internals_unit = time_matches.group(2)
            if time_internals_unit in ['s', 'ms']:
                return pa.time32(time_internals_unit)
            else:
                raise ValueError(f'{time_internals_unit} is not a valid unit for the pyarrow time32 type. Supported units: s (second) and ms (millisecond).')
        elif time_internals_bits == '64':
            time_internals_unit = time_matches.group(2)
            if time_internals_unit in ['us', 'ns']:
                return pa.time64(time_internals_unit)
            else:
                raise ValueError(f'{time_internals_unit} is not a valid unit for the pyarrow time64 type. Supported units: us (microsecond) and ns (nanosecond).')
        else:
            raise ValueError(_dtype_error_msg(datasets_dtype, 'time', examples=['time32[s]', 'time64[us]'], urls=['https://arrow.apache.org/docs/python/generated/pyarrow.time32.html', 'https://arrow.apache.org/docs/python/generated/pyarrow.time64.html']))
    decimal_matches = re.search('^decimal(.*)\\((.*)\\)$', datasets_dtype)
    if decimal_matches:
        decimal_internals_bits = decimal_matches.group(1)
        if decimal_internals_bits == '128':
            decimal_internals_precision_and_scale = re.search('^(\\d+),\\s*(-?\\d+)$', decimal_matches.group(2))
            if decimal_internals_precision_and_scale:
                precision = decimal_internals_precision_and_scale.group(1)
                scale = decimal_internals_precision_and_scale.group(2)
                return pa.decimal128(int(precision), int(scale))
            else:
                raise ValueError(_dtype_error_msg(datasets_dtype, 'decimal128', examples=['decimal128(10, 2)', 'decimal128(4, -2)'], urls=['https://arrow.apache.org/docs/python/generated/pyarrow.decimal128.html']))
        elif decimal_internals_bits == '256':
            decimal_internals_precision_and_scale = re.search('^(\\d+),\\s*(-?\\d+)$', decimal_matches.group(2))
            if decimal_internals_precision_and_scale:
                precision = decimal_internals_precision_and_scale.group(1)
                scale = decimal_internals_precision_and_scale.group(2)
                return pa.decimal256(int(precision), int(scale))
            else:
                raise ValueError(_dtype_error_msg(datasets_dtype, 'decimal256', examples=['decimal256(30, 2)', 'decimal256(38, -4)'], urls=['https://arrow.apache.org/docs/python/generated/pyarrow.decimal256.html']))
        else:
            raise ValueError(_dtype_error_msg(datasets_dtype, 'decimal', examples=['decimal128(12, 3)', 'decimal256(40, 6)'], urls=['https://arrow.apache.org/docs/python/generated/pyarrow.decimal128.html', 'https://arrow.apache.org/docs/python/generated/pyarrow.decimal256.html']))
    raise ValueError(f'Neither {datasets_dtype} nor {datasets_dtype + '_'} seems to be a pyarrow data type. Please make sure to use a correct data type, see: https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions')