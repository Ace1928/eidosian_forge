import pandas
import pandas._libs.lib as lib
from sqlalchemy import MetaData, Table, create_engine, inspect, text
from modin.core.storage_formats.pandas.parsers import _split_result_for_readers
def get_query_info(sql, con, partition_column):
    """
    Compute metadata needed for query distribution.

    Parameters
    ----------
    sql : str
        SQL query to be executed or a table name.
    con : SQLAlchemy connectable or str
        Database connection or url string.
    partition_column : str
        Column name used for data partitioning between the workers.

    Returns
    -------
    list
        Columns names list.
    str
        Query string.
    """
    engine = create_engine(con)
    if is_table(engine, sql):
        table_metadata = get_table_metadata(engine, sql)
        query = build_query_from_table(sql)
        cols = get_table_columns(table_metadata)
    else:
        check_query(sql)
        query = sql.replace(';', '')
        cols = get_query_columns(engine, query)
    return (list(cols.keys()), query)