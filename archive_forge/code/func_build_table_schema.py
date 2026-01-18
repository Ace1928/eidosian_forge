from pandas.core.common import (
def build_table_schema(data, index=True, primary_key=None, version=True):
    """
    Create a Table schema from ``data``.

    Parameters
    ----------
    data : Series, DataFrame
    index : bool, default True
        Whether to include ``data.index`` in the schema.
    primary_key : bool or None, default True
        column names to designate as the primary key.
        The default `None` will set `'primaryKey'` to the index
        level or levels if the index is unique.
    version : bool, default True
        Whether to include a field `pandas_version` with the version
        of pandas that generated the schema.

    Returns
    -------
    schema : dict

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {'A': [1, 2, 3],
    ...      'B': ['a', 'b', 'c'],
    ...      'C': pd.date_range('2016-01-01', freq='d', periods=3),
    ...     }, index=pd.Index(range(3), name='idx'))
    >>> build_table_schema(df)
    {'fields': [{'name': 'idx', 'type': 'integer'},
    {'name': 'A', 'type': 'integer'},
    {'name': 'B', 'type': 'string'},
    {'name': 'C', 'type': 'datetime'}],
    'pandas_version': '0.20.0',
    'primaryKey': ['idx']}

    Notes
    -----
    See `_as_json_table_type` for conversion types.
    Timedeltas as converted to ISO8601 duration format with
    9 decimal places after the secnods field for nanosecond precision.

    Categoricals are converted to the `any` dtype, and use the `enum` field
    constraint to list the allowed values. The `ordered` attribute is included
    in an `ordered` field.
    """
    if index is True:
        data = set_default_names(data)
    schema = {}
    fields = []
    if index:
        if data.index.nlevels > 1:
            for level in data.index.levels:
                fields.append(make_field(level))
        else:
            fields.append(make_field(data.index))
    if data.ndim > 1:
        for column, s in data.iteritems():
            fields.append(make_field(s))
    else:
        fields.append(make_field(data))
    schema['fields'] = fields
    if index and data.index.is_unique and (primary_key is None):
        if data.index.nlevels == 1:
            schema['primaryKey'] = [data.index.name]
        else:
            schema['primaryKey'] = data.index.names
    elif primary_key is not None:
        schema['primaryKey'] = primary_key
    if version:
        schema['pandas_version'] = '0.20.0'
    return schema