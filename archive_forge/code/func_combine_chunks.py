from typing import TYPE_CHECKING, List, Union
def combine_chunks(table: 'pyarrow.Table') -> 'pyarrow.Table':
    """This is pyarrow.Table.combine_chunks()
    with support for extension types.

    This will create a new table by combining the chunks the input table has.
    """
    from ray.air.util.transform_pyarrow import _concatenate_extension_column, _is_column_extension_type
    cols = table.columns
    new_cols = []
    for col in cols:
        if _is_column_extension_type(col):
            arr = _concatenate_extension_column(col)
        else:
            arr = col.combine_chunks()
        new_cols.append(arr)
    return pyarrow.Table.from_arrays(new_cols, schema=table.schema)