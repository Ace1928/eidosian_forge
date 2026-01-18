from typing import Dict
import numpy as np
import pandas
from modin.core.dataframe.pandas.interchange.dataframe_protocol.from_dataframe import (
from modin.tests.experimental.hdk_on_native.utils import ForceHdkImport
def get_data_of_all_types(has_nulls=False, exclude_dtypes=None, include_dtypes=None) -> Dict[str, np.ndarray]:
    """
    Generate a dictionary containing every datatype that is supported by HDK implementation of the exchange protocol.

    Parameters
    ----------
    has_nulls : bool, default: False
        Whether to include columns containing null values.
    exclude_dtypes : list, optional
        List of type prefixes to exclude in the dictionary. For example,
        passing ``["int", "float"]`` excludes all of the signed integer (``int16``,
         ``int32``, ``int64``) and float (``float32``, ``float64``) types.
    include_dtypes : list, optional
        List of type prefixes to include in the dictionary. For example,
        passing ``["int", "float"]`` will include ONLY signed integer (``int16``,
         ``int32``, ``int64``) and float (``float32``, ``float64``) types.

    Returns
    -------
    dict
        Dictionary to pass to a DataFrame constructor. The keys are string column names
        that are equal to the type name of the according column. Columns containing null
        types have a ``"_null"`` suffix in their names.
    """
    bool_data = {}
    int_data = {}
    uint_data = {}
    float_data = {}
    datetime_data = {}
    string_data = {}
    category_data = {}
    bool_data['bool'] = np.array([True, False, True, True] * 10, dtype=bool)
    for width in (8, 16, 32, 64):
        dtype = getattr(np, f'int{width}')
        max_val, min_val = (np.iinfo(dtype).max, np.iinfo(dtype).min)
        int_data[f'int{width}'] = np.array([max_val, max_val - 1, min_val + 1, min_val + 2] * 10, dtype=dtype)
    for width in (8, 16, 32, 64):
        dtype = getattr(np, f'uint{width}')
        max_val, min_val = (np.iinfo(dtype).max, np.iinfo(dtype).min)
        uint_data[f'uint{width}'] = np.array([max_val, max_val - 1, min_val + 1, min_val + 2] * 10, dtype=dtype)
    for width in (32, 64):
        dtype = getattr(np, f'float{width}')
        max_val, min_val = (np.finfo(dtype).max, np.finfo(dtype).min)
        float_data[f'float{width}'] = np.array([max_val, max_val - 1, min_val + 1, min_val + 2] * 10, dtype=dtype)
        if has_nulls:
            float_data[f'float{width}_null'] = np.array([max_val, None, min_val + 1, min_val + 2] * 10, dtype=dtype)
    for unit in ('s', 'ms', 'ns'):
        datetime_data[f'datetime64[{unit}]'] = np.array([0, 1, 2, 3] * 10, dtype=np.dtype(f'datetime64[{unit}]'))
        if has_nulls:
            datetime_data[f'datetime64[{unit}]_null'] = np.array([0, None, 2, 3] * 10, dtype=np.dtype(f'datetime64[{unit}]'))
    string_data['string'] = np.array(['English: test string', ' ', 'Chinese: 测试字符串', 'Russian: тестовая строка'] * 10)
    if has_nulls:
        string_data['string_null'] = np.array(['English: test string', None, 'Chinese: 测试字符串', 'Russian: тестовая строка'] * 10)
    category_data['category_string'] = pandas.Categorical(['Sample', 'te', ' ', 'xt'] * 10)
    if has_nulls:
        category_data['category_string_null'] = pandas.Categorical(['Sample', None, ' ', 'xt'] * 10)
    data = {**bool_data, **int_data, **uint_data, **float_data, **datetime_data, **string_data, **category_data}
    if include_dtypes is not None:
        filtered_keys = (key for key in data.keys() if any((key.startswith(dtype) for dtype in include_dtypes)))
        data = {key: data[key] for key in filtered_keys}
    if exclude_dtypes is not None:
        filtered_keys = (key for key in data.keys() if not any((key.startswith(dtype) for dtype in exclude_dtypes)))
        data = {key: data[key] for key in filtered_keys}
    return data