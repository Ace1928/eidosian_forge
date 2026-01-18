from __future__ import annotations
from typing import (
import warnings
from pandas._libs import lib
from pandas._libs.json import ujson_loads
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas import DataFrame
import pandas.core.common as com
from pandas.tseries.frequencies import to_offset
def convert_json_field_to_pandas_type(field) -> str | CategoricalDtype:
    """
    Converts a JSON field descriptor into its corresponding NumPy / pandas type

    Parameters
    ----------
    field
        A JSON field descriptor

    Returns
    -------
    dtype

    Raises
    ------
    ValueError
        If the type of the provided field is unknown or currently unsupported

    Examples
    --------
    >>> convert_json_field_to_pandas_type({"name": "an_int", "type": "integer"})
    'int64'

    >>> convert_json_field_to_pandas_type(
    ...     {
    ...         "name": "a_categorical",
    ...         "type": "any",
    ...         "constraints": {"enum": ["a", "b", "c"]},
    ...         "ordered": True,
    ...     }
    ... )
    CategoricalDtype(categories=['a', 'b', 'c'], ordered=True, categories_dtype=object)

    >>> convert_json_field_to_pandas_type({"name": "a_datetime", "type": "datetime"})
    'datetime64[ns]'

    >>> convert_json_field_to_pandas_type(
    ...     {"name": "a_datetime_with_tz", "type": "datetime", "tz": "US/Central"}
    ... )
    'datetime64[ns, US/Central]'
    """
    typ = field['type']
    if typ == 'string':
        return 'object'
    elif typ == 'integer':
        return field.get('extDtype', 'int64')
    elif typ == 'number':
        return field.get('extDtype', 'float64')
    elif typ == 'boolean':
        return field.get('extDtype', 'bool')
    elif typ == 'duration':
        return 'timedelta64'
    elif typ == 'datetime':
        if field.get('tz'):
            return f'datetime64[ns, {field['tz']}]'
        elif field.get('freq'):
            offset = to_offset(field['freq'])
            freq_n, freq_name = (offset.n, offset.name)
            freq = freq_to_period_freqstr(freq_n, freq_name)
            return f'period[{freq}]'
        else:
            return 'datetime64[ns]'
    elif typ == 'any':
        if 'constraints' in field and 'ordered' in field:
            return CategoricalDtype(categories=field['constraints']['enum'], ordered=field['ordered'])
        elif 'extDtype' in field:
            return registry.find(field['extDtype'])
        else:
            return 'object'
    raise ValueError(f'Unsupported or invalid field type: {typ}')