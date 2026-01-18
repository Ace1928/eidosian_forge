from pandas.core.common import (
def as_json_table_type(x):
    """
    Convert a NumPy / pandas type to its corresponding json_table.

    Parameters
    ----------
    x : array or dtype

    Returns
    -------
    t : str
        the Table Schema data types

    Notes
    -----
    This table shows the relationship between NumPy / pandas dtypes,
    and Table Schema dtypes.

    ==============  =================
    Pandas type     Table Schema type
    ==============  =================
    int64           integer
    float64         number
    bool            boolean
    datetime64[ns]  datetime
    timedelta64[ns] duration
    object          str
    categorical     any
    =============== =================
    """
    if is_integer_dtype(x):
        return 'integer'
    elif is_bool_dtype(x):
        return 'boolean'
    elif is_numeric_dtype(x):
        return 'number'
    elif is_datetime64_dtype(x) or is_datetime64tz_dtype(x):
        return 'datetime'
    elif is_timedelta64_dtype(x):
        return 'duration'
    elif is_categorical_dtype(x):
        return 'any'
    elif is_string_dtype(x):
        return 'string'
    else:
        return 'any'