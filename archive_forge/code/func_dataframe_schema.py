import json
import textwrap
@classmethod
def dataframe_schema(cls, p, safe=False):
    schema = {'type': 'array'}
    if safe is True:
        msg = 'DataFrame is not guaranteed to be safe for serialization as the column dtypes are unknown'
        raise UnsafeserializableException(msg)
    if p.columns is None:
        schema['items'] = {'type': 'object'}
        return schema
    mincols, maxcols = (None, None)
    if isinstance(p.columns, int):
        mincols, maxcols = (p.columns, p.columns)
    elif isinstance(p.columns, tuple):
        mincols, maxcols = p.columns
    if isinstance(p.columns, int) or isinstance(p.columns, tuple):
        schema['items'] = {'type': 'object', 'minItems': mincols, 'maxItems': maxcols}
    if isinstance(p.columns, list) or isinstance(p.columns, set):
        literal_types = [{'type': el} for el in cls.json_schema_literal_types.values()]
        allowable_types = {'anyOf': literal_types}
        properties = {name: allowable_types for name in p.columns}
        schema['items'] = {'type': 'object', 'properties': properties}
    minrows, maxrows = (None, None)
    if isinstance(p.rows, int):
        minrows, maxrows = (p.rows, p.rows)
    elif isinstance(p.rows, tuple):
        minrows, maxrows = p.rows
    if minrows is not None:
        schema['minItems'] = minrows
    if maxrows is not None:
        schema['maxItems'] = maxrows
    return schema