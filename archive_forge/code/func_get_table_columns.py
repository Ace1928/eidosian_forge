import pandas
import pandas._libs.lib as lib
from sqlalchemy import MetaData, Table, create_engine, inspect, text
from modin.core.storage_formats.pandas.parsers import _split_result_for_readers
def get_table_columns(metadata):
    """
    Extract columns names and python types from the `metadata`.

    Parameters
    ----------
    metadata : sqlalchemy.sql.schema.Table
        Table metadata.

    Returns
    -------
    dict
        Dictionary with columns names and python types.
    """
    cols = dict()
    for col in metadata.c:
        name = str(col).rpartition('.')[2]
        cols[name] = col.type.python_type.__name__
    return cols